"""Model loading, startup assertions and the forward pass.

Everything that can be wrong about this service and still answer 200 is checked
here, at boot, in a message that names the env var pinning it. The order matters:
cheap file checks before a 470 MB digest, digests before a CUDA context, and the
golden splice fixture before the warmup forward — so the failure a wrong splice
would cause is reported as a wrong splice.

MXG-144.
"""
import hashlib
import json
import logging
import os

import numpy as np
import torch

from ceserve import SPLICE_FIXTURE
from ceserve.constants import (
    BACKBONE,
    BOS,
    EOS,
    GAINS,
    HEAD_EXTRA,
    LABELS,
    MIN_MAX_LEN,
    MODEL_ID,
    MODEL_SHA256,
    NUM_LABELS,
    PAD,
    PROFILE,
    SPLICE_VERSION,
    TOKENIZER_SHA256,
    TOKENIZER_VERSION,
    VOCAB_SIZE,
    model_metadata,
    serving_contract,
)
from ceserve.splice import assemble, decode_token_ids

log = logging.getLogger(__name__)

WEIGHTS_FILE = "model.safetensors"
TOKENIZER_FILE = "tokenizer.json"
CONFIG_FILE = "config.json"
REQUIRED_FILES = (CONFIG_FILE, WEIGHTS_FILE, TOKENIZER_FILE)

# The measured marginal cost of one candidate at max_len 192 on a T4 with 4
# pinned cores, plus the fixed cost — `report/pipeline_v2/ce_latency_t4_v1.md`,
# least-squares over the k-sweep at k >= 60, R^2 >= 0.9976. Used only to refuse
# an inconsistent (MAX_INPUTS_PER_REQUEST, REQUEST_BUDGET_S) pair at boot.
MS_PER_CANDIDATE = 0.654
MS_FIXED = 0.3

DTYPES = {"fp16": torch.float16, "fp32": torch.float32, "bf16": torch.bfloat16}


def checkpoint_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _native_bf16():
    """True only for a device with real bf16 tensor cores (compute capability >= 8.0).

    `torch.cuda.is_bf16_supported()` defaults to `including_emulation=True`, so on
    Turing (sm_75 -- the T4 this service is sized for) it answers True and the
    guard it was written for never fires. The process then starts happily, reports
    the same serving_contract an H100 would, and runs on EMULATED bf16: measured on
    a T4, 2.48 TFLOP/s against fp16's 25.50 and fp32's 4.46 -- 10x slower than fp16
    and slower than fp32, with no error anywhere. Silence is the whole problem, so
    this asks the question that has a useful answer.

    Copied from `splade-service/backend.py`, deliberately verbatim: the kwarg
    landed in torch 2.6, and older builds get the capability check directly.
    """
    try:
        return torch.cuda.is_bf16_supported(including_emulation=False)
    except TypeError:
        return torch.cuda.get_device_capability()[0] >= 8


# ------------------------------------------------------------- assertion 1 --

def resolve_model_dir(model_dir):
    """`ship_mxg84.sh` nests the run name inside the dated release directory.

    Pointing at the outer one raises a tokenizer backend error that names
    nothing (`pipeline/ce_serve_skew.py` records the same trap). Turn it into a
    message that names the path to use instead of a stack trace to interpret.
    """
    if not os.path.isdir(model_dir):
        raise RuntimeError(f"CE_MODEL_DIR={model_dir} is not a directory")
    missing = [f for f in REQUIRED_FILES
               if not os.path.isfile(os.path.join(model_dir, f))]
    if not missing:
        return model_dir
    inner = [
        os.path.join(model_dir, name)
        for name in sorted(os.listdir(model_dir))
        if os.path.isfile(os.path.join(model_dir, name, CONFIG_FILE))
    ]
    if len(inner) == 1:
        raise RuntimeError(
            f"CE_MODEL_DIR={model_dir} has no {CONFIG_FILE}, but {inner[0]} does. "
            "ship_mxg84.sh nests the run name inside the dated release "
            "directory; point CE_MODEL_DIR at the inner directory. (Pointing at "
            "the outer one raises a tokenizer backend error that names nothing.)"
        )
    raise RuntimeError(
        f"CE_MODEL_DIR={model_dir} is missing {missing}; expected a checkpoint "
        f"directory containing {list(REQUIRED_FILES)}"
    )


# ------------------------------------------------------- assertions 2, 3, 4 --

def assert_digests(model_dir):
    weights = os.path.join(model_dir, WEIGHTS_FILE)
    actual = checkpoint_sha256(weights)
    if actual != MODEL_SHA256:
        raise RuntimeError(
            f"checkpoint SHA mismatch: expected {MODEL_SHA256}, got {actual} "
            f"for {weights} (CE_MODEL_SHA256)"
        )
    tokenizer = os.path.join(model_dir, TOKENIZER_FILE)
    actual = checkpoint_sha256(tokenizer)
    if actual != TOKENIZER_SHA256:
        raise RuntimeError(
            f"tokenizer SHA mismatch: expected {TOKENIZER_SHA256}, got {actual} "
            f"for {tokenizer} (CE_TOKENIZER_SHA256). This MUST equal the "
            "indexer's CE_TOKENIZER_SHA256, or the stored ceTokenIds are ids "
            "from a different vocabulary and every score is quietly wrong."
        )
    if not TOKENIZER_VERSION.strip():
        raise RuntimeError(
            "CE_TOKENIZER_VERSION is empty. An unset variable would fail "
            "loudly; an empty one matches no candidate's ceTokenizerVersion, so "
            "every candidate would be skipped and the service would report "
            "healthy while reranking nothing."
        )


# ---------------------------------------------------- assertions 5, 6, 7, 11 --

def assert_config(config_dict, max_len):
    architectures = config_dict.get("architectures") or []
    if list(architectures) != [BACKBONE]:
        raise RuntimeError(
            f"config.json architectures={architectures}, expected [{BACKBONE!r}]"
        )
    if int(config_dict.get("vocab_size", -1)) != VOCAB_SIZE:
        raise RuntimeError(
            f"config.json vocab_size={config_dict.get('vocab_size')} != "
            f"{VOCAB_SIZE}; the stored ceTokenIds are ids in THIS vocabulary"
        )
    labels = config_dict.get("id2label") or {}
    n_labels = len(labels) if labels else int(config_dict.get("num_labels", 0))
    if n_labels != NUM_LABELS:
        raise RuntimeError(
            f"config.json declares {n_labels} labels, expected {NUM_LABELS} "
            f"({'/'.join(LABELS)})"
        )
    if n_labels != len(GAINS):
        # Unreachable given the check above, kept because the failure it guards
        # is silent: a 3-class head broadcasts against a 4-element gain vector
        # and produces a perfectly plausible number.
        raise RuntimeError(
            f"GAINS has {len(GAINS)} entries but the head has {n_labels} labels"
        )
    for field, expected in (("pad_token_id", PAD), ("bos_token_id", BOS),
                            ("eos_token_id", EOS)):
        actual = config_dict.get(field)
        if actual is None or int(actual) != expected:
            raise RuntimeError(
                f"config.json {field}={actual} but splice.assemble assumes "
                f"{field.split('_')[0].upper()}={expected}; the splice is not "
                "portable to this checkpoint — it would emit a syntactically "
                "valid, semantically wrong sequence"
            )
    positions = int(config_dict.get("max_position_embeddings", 0))
    # XLM-R offsets positions past `padding_idx`, so the usable width is
    # max_position_embeddings - 2 (514 -> 512).
    ceiling = positions - 2
    if not (MIN_MAX_LEN <= max_len <= ceiling):
        raise RuntimeError(
            f"CE_MAX_LEN={max_len} outside [{MIN_MAX_LEN}, {ceiling}] "
            f"(max_position_embeddings={positions}). Below {MIN_MAX_LEN} the "
            "clamp chain in assemble() can emit a row wider than max_len."
        )


# ------------------------------------------------------ assertions 8, 9, 10 --

def resolve_device(name, allow_cpu):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CE_DEVICE=cuda but torch.cuda.is_available() is False"
            )
        capability = torch.cuda.get_device_capability()
        if capability[0] < 7:
            raise RuntimeError(
                f"compute capability {capability} has no usable fp16 sdpa path; "
                "this service is sized for sm_75 (T4) or better"
            )
        return torch.device("cuda")
    if not allow_cpu:
        raise RuntimeError(
            "refusing to serve the CE on CPU: a 120-candidate window costs "
            "seconds there and would blow the 150 ms budget while every "
            "healthcheck stayed green. Set CE_ALLOW_CPU=1 only for tests."
        )
    return torch.device("cpu")


def resolve_dtype(name, device):
    if name not in DTYPES:
        raise RuntimeError(f"CE_DTYPE={name!r} must be one of {sorted(DTYPES)}")
    if device.type != "cuda":
        # Half precision on CPU is neither fast nor faithful; the CPU path only
        # exists for tests and for the fp32 parity cell of the agreement check.
        return "fp32", torch.float32
    if name == "bf16" and not _native_bf16():
        raise RuntimeError(
            "the selected CUDA device has no native bf16 (compute capability "
            f"{torch.cuda.get_device_capability(device)}); refusing to run on "
            "emulated bf16 at ~10x the cost -- pick fp16 or fp32 (CE_DTYPE)"
        )
    return name, DTYPES[name]


# ------------------------------------------------------------ assertion 12 --

def assert_budget_is_consistent(max_inputs, request_budget_s):
    worst_ms = MS_PER_CANDIDATE * max_inputs + MS_FIXED
    if worst_ms > request_budget_s * 1000:
        raise RuntimeError(
            f"MAX_INPUTS_PER_REQUEST={max_inputs} needs ~{worst_ms:.0f} ms at "
            f"the measured {MS_PER_CANDIDATE} ms/candidate, but "
            f"REQUEST_BUDGET_S={request_budget_s} allows "
            f"{request_budget_s * 1000:.0f} ms. embedding-service shipped "
            "exactly this pair (MAX_INPUTS=256 advertised against a budget "
            "nothing could finish inside) and every legal request timed out, "
            "which marked its only backend unhealthy and took the dense path "
            "down. The two knobs are not independent."
        )


# ------------------------------------------------------------ assertion 13 --

def assert_golden_fixture(path=SPLICE_FIXTURE):
    """Re-run the frozen splice on the serving box, before the first request.

    ~1 ms, and it is the difference between "CI was green on some other machine"
    and "the code in this container splices correctly". A wrong splice produces
    plausible scores, not an error, so this is the only check that can catch a
    bad merge, a bad bind-mount or a stale image.
    """
    payload = json.loads(path.read_text())
    if payload.get("spliceVersion") != SPLICE_VERSION:
        raise RuntimeError(
            f"golden fixture spliceVersion={payload.get('spliceVersion')!r} but "
            f"this build is {SPLICE_VERSION!r}; regenerate with "
            "pipeline/gen_ce_splice_fixture.py"
        )
    if payload.get("tokenizerSha256") != TOKENIZER_SHA256:
        raise RuntimeError(
            "golden fixture was built from a different tokenizer "
            f"({payload.get('tokenizerSha256')} != {TOKENIZER_SHA256})"
        )
    rows = 0
    for case in payload["cases"]:
        q_ids = np.asarray(case["queryIds"], dtype=np.int64)
        arts = [decode_token_ids(a["tokensB64"], a["tokenCount"])
                for a in case["articles"]]
        ids, mask, max_seq = assemble(q_ids, arts, case["maxLen"], None)
        if (ids.tolist() != case["expectedIds"]
                or mask.tolist() != case["expectedMask"]
                or max_seq != case["expectedMaxSeq"]):
            raise RuntimeError(
                f"GOLDEN SPLICE MISMATCH on case {case['name']!r}: "
                "splice.assemble does not reproduce the frozen fixture. A wrong "
                "splice produces plausible scores rather than an error, so this "
                "process refuses to serve. See tests/test_ce_splice_parity.py."
            )
        rows += len(arts)
    return len(payload["cases"]), rows


# -------------------------------------------------------------- the scorer --

def ce_score(probs):
    """`sum(softmax(logits) * GAINS) / 4` — `train_ce.py`'s `ce_score`.

    ONE implementation, imported by the service and by the agreement test, so
    the two cannot be comparing different quantities.
    """
    return (np.asarray(probs, dtype=np.float64) * GAINS).sum(axis=-1) / 4.0


class CrossEncoderScorer:
    def __init__(self, config):
        self.model_dir = resolve_model_dir(config.model_dir)
        assert_digests(self.model_dir)

        with open(os.path.join(self.model_dir, CONFIG_FILE)) as handle:
            config_dict = json.load(handle)
        assert_config(config_dict, config.max_len)

        self.device = resolve_device(config.device, config.allow_cpu)
        self.dtype_name, self.dtype = resolve_dtype(config.dtype, self.device)
        assert_budget_is_consistent(config.max_inputs, config.request_budget_s)
        if config.assert_golden:
            cases, rows = assert_golden_fixture()
            log.info("golden splice fixture ok: %d cases / %d rows", cases, rows)

        if config.torch_num_threads > 0:
            torch.set_num_threads(config.torch_num_threads)

        # Deliberately NOT AutoTokenizer/AutoModel.from_pretrained:
        #  * `Tokenizer.from_file` skips the slow-tokenizer conversion path, so
        #    sentencepiece and protobuf stay out of the serving runtime (~60 MB
        #    less to bind-mount);
        #  * building from config.json and loading the state dict with
        #    strict=True turns a renamed or missing tensor into a hard error.
        #    `from_pretrained` warns and randomly initialises, which would serve
        #    a plausible-looking model with an untrained head.
        from tokenizers import Tokenizer
        from transformers import XLMRobertaConfig, XLMRobertaForSequenceClassification
        from safetensors.torch import load_file

        self.tokenizer = Tokenizer.from_file(
            os.path.join(self.model_dir, TOKENIZER_FILE))
        # The Rust backend is stateful and shared. Loading from file gives a
        # clean one, but be explicit: a stale padding config makes the model
        # attend PAD with no error anywhere (report/ce_distill_v1.md).
        self.tokenizer.no_padding()
        self.tokenizer.no_truncation()
        if self.tokenizer.get_vocab_size() > VOCAB_SIZE:
            raise RuntimeError(
                f"tokenizer vocab {self.tokenizer.get_vocab_size()} exceeds the "
                f"model's {VOCAB_SIZE}"
            )

        model = XLMRobertaForSequenceClassification(
            XLMRobertaConfig.from_dict(config_dict))
        model.load_state_dict(
            load_file(os.path.join(self.model_dir, WEIGHTS_FILE), device="cpu"),
            strict=True,
        )
        self.model = model.to(device=self.device, dtype=self.dtype).eval()

        self.max_len = int(config.max_len)
        self.chunk = int(config.forward_chunk)
        self.attn_implementation = getattr(
            self.model.config, "_attn_implementation", "sdpa")
        self.degraded = False
        self.warmed_up = False

    # ------------------------------------------------------------ warmup --

    def warmup(self):
        """A synthetic k=8 window at the serving width, before /readyz goes green.

        MXG-111's lesson, encoded: `splade-backend` kept `/health`, `/readyz` and
        `/embed` all green while `/encode-packed` returned 500, because nothing
        exercised the path that was actually broken. A healthcheck that does not
        run a forward is a healthcheck for the HTTP server, not the model.
        """
        rng = np.random.default_rng(0)
        q_ids = rng.integers(4, VOCAB_SIZE, size=8, dtype=np.int64)
        arts = [rng.integers(4, VOCAB_SIZE, size=self.max_len, dtype=np.int64)
                for _ in range(8)]
        probs, _ = self.score(q_ids, arts, self.max_len)
        if probs.shape != (8, len(LABELS)) or not np.isfinite(probs).all():
            raise RuntimeError(
                f"warmup forward produced {probs.shape} / finite="
                f"{bool(np.isfinite(probs).all())}, expected (8, {len(LABELS)}) "
                "and all-finite"
            )
        self.warmed_up = True
        return probs.shape

    # ------------------------------------------------------------- score --

    def score(self, q_ids, arts, max_len):
        """(probs[n, 4] float32, stats). Chunks are taken IN REQUEST ORDER.

        Sorting by length would shrink the padded width, and it is tempting. Two
        reasons not to: the 0.654 ms/candidate the budget is built on was
        measured unsorted on real windows, and a sorted chunk can end up with
        every row the same length, which makes the attention mask all ones,
        which lets sdpa drop the mask and dispatch a different kernel. That is
        the same rounding hazard `splade-service`'s `pad_guard` exists for, and
        it would be introduced silently by a "harmless" optimisation.
        """
        out, widths = [], []
        for start in range(0, len(arts), self.chunk):
            ids, mask, _ = assemble(
                q_ids, arts[start:start + self.chunk], max_len, None)
            widths.append(int(ids.shape[1]))
            input_ids = torch.from_numpy(ids).to(self.device, non_blocking=True)
            attention = torch.from_numpy(mask).to(self.device, non_blocking=True)
            # inference_mode, not no_grad: without it PyTorch keeps the FFN
            # intermediate activations alive across chunks, and a tight queue
            # OOMs a card this service does not own alone.
            with torch.inference_mode():
                logits = self.model(input_ids=input_ids,
                                    attention_mask=attention).logits
                out.append(torch.softmax(logits.float(), -1).cpu().numpy())
        probs = np.concatenate(out) if out else np.zeros((0, len(LABELS)),
                                                         dtype=np.float32)
        return probs, {"padded_width": max(widths) if widths else 0,
                       "chunks": len(widths)}

    # ---------------------------------------------------------- metadata --

    def metadata(self):
        payload = dict(model_metadata(self.dtype_name, self.max_len))
        payload.update({
            "device": str(self.device),
            "dtype": self.dtype_name,
            "attn_implementation": self.attn_implementation,
            "forward_chunk": self.chunk,
            "torch_version": torch.__version__,
            "warmed_up": self.warmed_up,
            "degraded": self.degraded,
        })
        if self.device.type == "cuda":
            index = self.device.index or 0
            payload["gpu_name"] = torch.cuda.get_device_name(index)
            payload["gpu_capability"] = list(
                torch.cuda.get_device_capability(index))
        return payload

    @property
    def serving_contract(self):
        return serving_contract(self.dtype_name, self.max_len)


__all__ = [
    "CrossEncoderScorer",
    "MS_FIXED",
    "MS_PER_CANDIDATE",
    "assert_budget_is_consistent",
    "assert_config",
    "assert_digests",
    "assert_golden_fixture",
    "ce_score",
    "checkpoint_sha256",
    "resolve_device",
    "resolve_dtype",
    "resolve_model_dir",
    "HEAD_EXTRA",
    "MODEL_ID",
    "PROFILE",
]
