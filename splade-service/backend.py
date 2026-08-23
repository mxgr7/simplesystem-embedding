import asyncio
import contextlib
import hashlib
import json
import os
import re
import secrets
import struct
import threading
import unicodedata
from collections import deque
from contextlib import asynccontextmanager

import numpy as np
import torch
import transformers
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from huggingface_hub import hf_hub_download
from pydantic import BaseModel
from tokenizers.implementations import BertWordPieceTokenizer
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from constants import (
    MAX_OFFER_LENGTH,
    MODEL_NAME,
    MODEL_SHA256,
    TOP_K,
    model_metadata,
)
from codec import pack_sparse_rows


class EncodeRequest(BaseModel):
    inputs: list[str]
    document: bool = True


def is_cased_token(tok):
    """True if the token string carries uppercase or a diacritic -- i.e. text
    folding (lowercase + strip diacritics) would change it.

    Copied verbatim from embedding_train.splade_model.is_cased_token: the mask it
    builds is part of the trained model's contract, so the two implementations
    have to agree exactly or the served vector is not the trained one.
    """
    s = tok.replace("##", "")
    if re.search(r"[A-ZÄÖÜ]", s):
        return True
    return any(unicodedata.combining(c) for c in unicodedata.normalize("NFD", s))


def checkpoint_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _native_bf16():
    """True only for a device with real bf16 tensor cores (compute capability >= 8.0).

    `torch.cuda.is_bf16_supported()` defaults to `including_emulation=True`, so on
    Turing (sm_75 -- the T4 that hosts the dense embedding service) it answers True
    and the guard it was written for never fires. The backend then starts happily,
    reports the same `document_encoding_version` an H100 would, and encodes on
    EMULATED bf16: measured on a T4, 2.48 TFLOP/s against fp16's 25.50 and fp32's
    4.46 -- 10x slower than fp16 and slower than fp32, with no error anywhere.
    Silence is the whole problem, so this asks the question that has a useful answer.

    The kwarg landed in torch 2.6; older builds get the capability check directly.
    """
    try:
        return torch.cuda.is_bf16_supported(including_emulation=False)
    except TypeError:
        return torch.cuda.get_device_capability()[0] >= 8


def build_vocab_mask(tokenizer, vocab_size, model_hp):
    """The trained model's output mask, rebuilt from its own hyper-parameters.

    Byte-for-byte the same construction as
    `embedding_train.splade_model.SpladeModule.__init__`: special ids, then the
    optional train-time stopword stoplist, then the optional folded-vocab (cased /
    diacritic) mask. Returns the boolean keep-mask plus a detail dict; the sha256
    of the mask bits goes into /metadata so a client can pin the exact mask rather
    than trusting a pair of booleans to describe it.
    """
    mask = torch.ones(vocab_size, dtype=torch.bool)
    special = [i for i in tokenizer.all_special_ids if 0 <= i < vocab_size]
    mask[special] = False
    stopword_ids = [int(i) for i in (model_hp.get("stopword_mask_ids") or [])
                    if 0 <= int(i) < vocab_size]
    if stopword_ids:
        mask[stopword_ids] = False
    fold = bool(model_hp.get("fold_vocab_mask", False))
    cased = []
    if fold:
        # Index by the REAL token id, not enumeration order, so added tokens or
        # gaps cannot misalign the mask -- same reasoning as the training code.
        cased = [token_id for token, token_id in tokenizer.get_vocab().items()
                 if token_id < vocab_size and is_cased_token(token)]
        mask[cased] = False
    detail = {
        "fold_vocab_mask": fold,
        "cased_masked": len(cased),
        "stopword_masked": len(stopword_ids),
        "special_masked": len(special),
        "kept_dims": int(mask.sum().item()),
        "vocab_mask_sha256": hashlib.sha256(
            mask.numpy().tobytes()).hexdigest(),
    }
    return mask, detail


def load_tokenizer(model_name, cache_dir):
    try:
        return AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    except (ImportError, ValueError, OSError):
        vocab_path = hf_hub_download(
            model_name, "vocab.txt", cache_dir=cache_dir
        )
        lower = False
        try:
            config_path = hf_hub_download(
                model_name, "tokenizer_config.json", cache_dir=cache_dir
            )
            with open(config_path) as handle:
                lower = bool(json.load(handle).get("do_lower_case", False))
        except Exception:
            pass
        tokenizer = BertWordPieceTokenizer(
            vocab_path,
            lowercase=lower,
            strip_accents=lower,
        )
        return PreTrainedTokenizerFast(
            tokenizer_object=tokenizer._tokenizer,
            unk_token="[UNK]",
            sep_token="[SEP]",
            pad_token="[PAD]",
            cls_token="[CLS]",
            mask_token="[MASK]",
            model_max_length=512,
        )


_HEAD = struct.Struct("<H")


def pack_rows_vectorised(ids, values):
    """[B,K] int32 ids + [B,K] float16 weights -> list of B wire rows.

    One sort and one buffer build for the whole batch instead of the per-row
    numpy work `pack_sparse_arrays` does. Output is byte-identical to
    `[pack_sparse_arrays(i, v) for i, v in zip(ids, values)]`: non-positive and
    non-finite entries dropped, survivors ordered by ascending token id.
    """
    positive = np.isfinite(values) & (values > 0)
    counts = positive.sum(1)
    # push dropped entries past every real id so they land after the survivors
    order = np.argsort(np.where(positive, ids, np.int32(1 << 30)), axis=1, kind="stable")
    sorted_ids = np.take_along_axis(ids, order, 1).astype("<u2")
    sorted_values = np.take_along_axis(values, order, 1).astype("<f2")
    rows = []
    for row in range(len(counts)):
        count = int(counts[row])
        rows.append(
            _HEAD.pack(count)
            + sorted_ids[row, :count].tobytes()
            + sorted_values[row, :count].tobytes()
        )
    return rows


class SpladeEncoder:
    def __init__(self, checkpoint_path, device, cache_dir, batch_size, document_dtype="bf16",
                 weights_cast=False, compile_head=False, overlap=2, pad_guard=0,
                 fold_vocab_mask=None):
        actual_sha = checkpoint_sha256(checkpoint_path)
        if actual_sha != MODEL_SHA256:
            raise RuntimeError(
                f"checkpoint SHA mismatch: expected {MODEL_SHA256}, got {actual_sha}"
            )
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
        hyperparameters = checkpoint["hyper_parameters"]
        if hyperparameters["model"]["model_name"] != MODEL_NAME:
            raise RuntimeError("checkpoint backbone does not match service contract")

        config_dict, _ = transformers.PretrainedConfig.get_config_dict(
            MODEL_NAME, cache_dir=cache_dir
        )
        architecture = config_dict["architectures"][0]
        model_class = getattr(transformers, architecture)
        config = model_class.config_class.from_dict(config_dict)
        model = model_class(config)
        state_dict = {
            key.removeprefix("encoder."): value
            for key, value in checkpoint["state_dict"].items()
        }
        model.load_state_dict(state_dict, strict=True)

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        dtype_names = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": None}
        self.document_dtype_name = document_dtype if self.device.type == "cuda" else "fp32"
        if self.document_dtype_name not in dtype_names:
            raise ValueError("DOCUMENT_DTYPE must be bf16, fp16, or fp32")
        self.document_dtype = dtype_names[self.document_dtype_name]
        if (
            self.device.type == "cuda"
            and self.document_dtype_name == "bf16"
            and not _native_bf16()
        ):
            raise ValueError(
                "the selected CUDA device has no native bf16 (compute capability "
                f"{torch.cuda.get_device_capability(self.device)}); refusing to run "
                "on emulated bf16 -- pick fp16 or fp32"
            )

        # `weights_cast` holds the whole model in the compute dtype instead of
        # casting fp32 weights on every matmul. That also puts LayerNorm in bf16,
        # which is where the measured win actually comes from -- and which is why
        # it is NOT bit-exact. It was gated on the gold/seg recall harness before
        # being made available here (see report/pipeline_v2/splade_encode_throughput.md).
        self.weights_cast = bool(weights_cast) and self.document_dtype is not None
        weight_dtype = self.document_dtype if self.weights_cast else torch.float32
        self.model = model.to(device=self.device, dtype=weight_dtype).eval()
        self.bert = getattr(self.model, "bert", None)
        self.head = getattr(self.model, "cls", None)
        self.tokenizer = load_tokenizer(MODEL_NAME, cache_dir)
        # The output mask is READ FROM THE CHECKPOINT, not configured. Training
        # multiplies every representation by `special_token_vocab_mask`, so a
        # dimension it zeroes never receives a gradient and the L1/FLOPS
        # regularizer never prices it -- its activation is arbitrary. Serving the
        # same checkpoint without the same mask is not a tuning difference: for
        # `soup2b50` (fold_vocab_mask=True) the query vector goes from 5.1 to
        # ~4,530 non-zeros, i.e. 4,530 posting lists per query instead of 5.
        #
        # Deriving it from `hyper_parameters.model` rather than from the
        # environment is the point of this code: a hand-set flag is exactly the
        # kind of thing that silently disagrees with the weights it is serving.
        # Mirrors embedding_train.splade_model.SpladeModule.__init__.
        model_hp = hyperparameters["model"]
        self.mask, self.mask_detail = build_vocab_mask(
            self.tokenizer, config.vocab_size, model_hp
        )
        self.fold_vocab_mask = self.mask_detail["fold_vocab_mask"]
        self.cased_masked = self.mask_detail["cased_masked"]
        # An operator override may only AGREE with the checkpoint. Disagreeing is
        # a hard error rather than a silent win for either side, because both
        # directions produce a plausible-looking service that serves the wrong
        # vectors.
        if fold_vocab_mask is not None and bool(fold_vocab_mask) != self.fold_vocab_mask:
            raise RuntimeError(
                f"SPLADE_FOLD_VOCAB_MASK={fold_vocab_mask} contradicts the "
                f"checkpoint's model.fold_vocab_mask={self.fold_vocab_mask}; "
                f"the mask is part of the trained model, not a serving option"
            )
        self.mask = self.mask.to(self.device)
        self.drop = (~self.mask).to(self.device)
        self.pad_token_id = self.tokenizer.pad_token_id or 0
        self.batch_size = batch_size
        self.pad_guard = int(pad_guard)

        # Compiling the MLM head re-associates its GEMM+reduction, so it is not
        # bit-exact either; same recall gate as weights_cast. `bert` cannot be
        # compiled at all -- BertModel.forward and BertForMaskedLM.forward both
        # carry a @capture_outputs decorator that raises
        # "NameError: name 'torch' is not defined" under dynamo.
        self.compile_head = bool(compile_head) and self.device.type == "cuda"
        self.split_call = self.compile_head and self.bert is not None
        if self.compile_head:
            if self.head is None:
                raise RuntimeError("cannot compile head: model has no `cls` module")
            self.head = torch.compile(self.head, mode="default", dynamic=True)

        self.overlap = max(int(overlap), 1) if self.device.type == "cuda" else 1
        # `encode_packed` runs in a worker thread (asyncio.to_thread) and the
        # service may have several requests in flight, so the copy stream and the
        # pinned staging ring must not be shared between them. Threads are reused
        # by the executor, so the pinned buffers are still allocated once each.
        self._local = threading.local()

        self.optimized_document_encoder = self.device.type == "cuda"
        self.encoder_implementation = "pytorch-cuda-sorted-maxsplit-packed-v2"
        variant = "" if (self.weights_cast or self.compile_head) else "-exact"
        if self.weights_cast:
            variant += "-wcast"
        if self.compile_head:
            variant += "-chead"
        if self.pad_guard:
            variant += "-padguard"
        self.document_encoding_version = (
            f"prod-soup-top256-{self.document_dtype_name}-fp16codec{variant}-v2"
        )

    def encode(self, texts, document=True):
        """The JSON transport -- and, `document=False`, the query path.

        Runs the same forward as `_encode_batch`: same `_autocast()`, same
        `_logits()`, same max-first in-place activation. That is not tidiness.
        `document_encoding_version` names the compute dtype and the fast-path
        flags; it namespaces the KVRocks keyspace and it is one of the four keys
        every backend in the pool is gated on. And this is not the minor
        transport -- `BackendPool` -> `/encode` is what `/embed` uses, which is
        what the indexer uses. A `/encode` that ignored `DOCUMENT_DTYPE` would
        file fp32 vectors under a name that says bf16, into the same cache the
        packed transport writes its bf16 ones to, with nothing downstream able
        to tell the two apart.

        Queries share the forward deliberately: a query vector and a document
        vector meet in a dot product, so they have to come out of the same
        arithmetic. They already did whenever `DOCUMENT_WEIGHTS_CAST` held the
        weights in the compute dtype -- which is the live T4 -- so this only
        extends that to the autocast profiles.
        """
        output = []
        for start in range(0, len(texts), self.batch_size):
            chunk = texts[start:start + self.batch_size]
            tokens = self.tokenizer(
                chunk,
                padding=True,
                truncation=True,
                max_length=MAX_OFFER_LENGTH,
                return_tensors="pt",
            )
            tokens = {
                name: value.to(self.device) for name, value in tokens.items()
            }
            with torch.inference_mode(), self._autocast():
                logits = self._logits(
                    tokens["input_ids"], tokens["attention_mask"]
                )
                # Max first, then log1p(relu(.)) -- monotone non-decreasing, so
                # this is the vector `_encode_batch` computes, elementwise.
                logits.masked_fill_(
                    ~tokens["attention_mask"].bool().unsqueeze(-1), float("-inf")
                )
                vectors = logits.amax(dim=1)
                # in-place, NOT torch.log1p/torch.relu: log1p carries an fp32
                # autocast cast policy, so the functional form silently upcasts
                # the result -- which would put this transport back on a
                # different number from the packed one even under one autocast.
                vectors.relu_().log1p_()
                vectors.masked_fill_(self.drop, 0)

            if document:
                values, ids = torch.topk(vectors, TOP_K, dim=1, sorted=True)
                for row_ids, row_values in zip(ids, values):
                    positive = row_values > 0
                    output.append({
                        str(int(token_id)): float(weight)
                        for token_id, weight in zip(
                            row_ids[positive].cpu(), row_values[positive].cpu()
                        )
                    })
            else:
                for vector in vectors:
                    ids = torch.nonzero(vector > 0).flatten()
                    output.append({
                        str(int(token_id)): float(vector[token_id])
                        for token_id in ids.cpu()
                    })
            del logits, vectors
        return output

    def _autocast(self):
        if self.document_dtype is None or self.weights_cast:
            return contextlib.nullcontext()
        return torch.autocast(self.device.type, dtype=self.document_dtype)

    def _logits(self, input_ids, attention_mask):
        if self.split_call:
            hidden = self.bert(
                input_ids=input_ids, attention_mask=attention_mask
            ).last_hidden_state
            return self.head(hidden)
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits

    @torch.inference_mode()
    def _encode_batch(self, input_ids, attention_mask, min_len):
        """[B,L] -> (values [B,K] f16, ids [B,K] i32), both still on the GPU.

        `log1p(relu(.))` is monotone non-decreasing and >= 0, so
            amax_L( f(x) * mask ) == f( amax_L( x masked with -inf ) )
        Taking the max FIRST runs relu/log1p over [B,V] instead of [B,L,V] -- at
        B=256, L=256 that head output is 4 GB, and this removes ~6 of 8 passes
        over it. Because the batch is length-sorted, only the ragged tail past
        the shortest row needs masking at all.
        """
        with self._autocast():
            logits = self._logits(input_ids, attention_mask)
            length = logits.shape[1]
            lmin = max(min(int(min_len), length), 1)
            vectors = logits.narrow(1, 0, lmin).amax(dim=1)  # read-only, no mask needed
            if length > lmin:
                tail = logits.narrow(1, lmin, length - lmin)
                tail.masked_fill_(
                    ~attention_mask[:, lmin:].bool().unsqueeze(-1), float("-inf")
                )
                torch.maximum(vectors, tail.amax(dim=1), out=vectors)
            # in-place, NOT torch.log1p/torch.relu: log1p carries an fp32 autocast
            # cast policy, so the functional form silently upcasts the result and
            # breaks bit-equivalence with the reference encoder.
            vectors.relu_().log1p_()
            vectors.masked_fill_(self.drop, 0)
            values, ids = torch.topk(vectors.float(), TOP_K, dim=1, sorted=True)
        return values.to(torch.float16), ids.to(torch.int32)

    def _batches(self, lengths, order, token_lists):
        """Length-sorted, dynamically padded batches over the whole request."""
        limit = int(getattr(self.tokenizer, "model_max_length", 512) or 512)
        for start in range(0, len(order), self.batch_size):
            selection = order[start:start + self.batch_size]
            batch_lengths = lengths[selection]
            width = int(batch_lengths.max())
            if self.pad_guard and int(batch_lengths.min()) >= width:
                # A batch with NO padding has an all-ones attention mask, and the
                # sdpa path then drops the mask and calls a different kernel --
                # same maths, different rounding. Length sorting groups the ~37%
                # of documents that sit exactly at the truncation cap into
                # precisely such batches. One extra pad column keeps every batch
                # on the masked kernel, for ~1/L extra compute.
                width = min(width + self.pad_guard, limit)
            input_ids = np.full(
                (len(selection), width), self.pad_token_id, dtype=np.int64
            )
            attention = np.zeros((len(selection), width), dtype=np.int64)
            for row, index in enumerate(selection):
                tokens = token_lists[index]
                input_ids[row, :len(tokens)] = tokens
                attention[row, :len(tokens)] = 1
            yield selection, input_ids, attention, int(batch_lengths.min())

    def _copy_stream(self):
        if self.device.type != "cuda":
            return None
        stream = getattr(self._local, "copy_stream", None)
        if stream is None:
            stream = torch.cuda.Stream(device=self.device)
            self._local.copy_stream = stream
        return stream

    def _staging_slot(self, index):
        """Ring of pinned [batch,K] buffers so the D2H copy can be async."""
        staging = getattr(self._local, "staging", None)
        if staging is None:
            staging = []
            self._local.staging = staging
        depth = self.overlap + 1
        while len(staging) < depth:
            staging.append((
                torch.empty((self.batch_size, TOP_K), dtype=torch.float16, pin_memory=True),
                torch.empty((self.batch_size, TOP_K), dtype=torch.int32, pin_memory=True),
            ))
        return staging[index % depth]

    def encode_packed(self, texts):
        encoded = self.tokenizer(
            texts, truncation=True, max_length=MAX_OFFER_LENGTH
        )["input_ids"]
        lengths = np.fromiter(
            (len(tokens) for tokens in encoded), dtype=np.int64, count=len(encoded)
        )
        order = np.argsort(lengths, kind="stable")
        rows = [None] * len(texts)
        pending = deque()
        copy_stream = self._copy_stream()

        def drain():
            done, values_pinned, ids_pinned, selection = pending.popleft()
            done.synchronize()
            count = len(selection)
            packed = pack_rows_vectorised(
                ids_pinned[:count].numpy(), values_pinned[:count].numpy()
            )
            for position, row in zip(selection, packed):
                rows[position] = row

        slot = 0
        for selection, input_ids, attention, min_len in self._batches(
            lengths, order, encoded
        ):
            device_ids = torch.from_numpy(input_ids).to(self.device, non_blocking=True)
            device_attention = torch.from_numpy(attention).to(
                self.device, non_blocking=True
            )
            values, ids = self._encode_batch(device_ids, device_attention, min_len)

            if copy_stream is None:
                packed = pack_rows_vectorised(ids.numpy(), values.numpy())
                for position, row in zip(selection, packed):
                    rows[position] = row
                continue

            # Hand the D2H copy to a second stream so the next batch's forward
            # starts immediately instead of waiting on the transfer.
            while len(pending) >= self.overlap:
                drain()
            values_pinned, ids_pinned = self._staging_slot(slot)
            slot += 1
            ready = torch.cuda.Event()
            ready.record()
            with torch.cuda.stream(copy_stream):
                copy_stream.wait_event(ready)
                count = len(selection)
                values_pinned[:count].copy_(values, non_blocking=True)
                ids_pinned[:count].copy_(ids, non_blocking=True)
                done = torch.cuda.Event()
                done.record(copy_stream)
            # the caching allocator must not hand these buffers to another stream
            # until the copy stream is finished reading them
            values.record_stream(copy_stream)
            ids.record_stream(copy_stream)
            pending.append((done, values_pinned, ids_pinned, selection))

        while pending:
            drain()
        return pack_sparse_rows(rows)


@asynccontextmanager
async def lifespan(app):
    checkpoint = os.environ.get("SPLADE_CHECKPOINT", "/model/prod_soup.ckpt")
    device = os.environ.get("DEVICE", "auto")
    cache_dir = os.environ.get("HF_HOME", "/data/huggingface")
    batch_size = int(os.environ.get("BACKEND_BATCH_SIZE", "256"))
    document_dtype = os.environ.get("DOCUMENT_DTYPE", "bf16").lower()
    threads = int(os.environ.get("TORCH_NUM_THREADS", str(os.cpu_count() or 1)))
    torch.set_num_threads(threads)
    truthy = {"1", "true", "yes", "on"}
    app.state.encoder = await asyncio.to_thread(
        SpladeEncoder,
        checkpoint,
        device,
        cache_dir,
        batch_size,
        document_dtype,
        os.environ.get("DOCUMENT_WEIGHTS_CAST", "0").lower() in truthy,
        os.environ.get("DOCUMENT_COMPILE_HEAD", "0").lower() in truthy,
        int(os.environ.get("DOCUMENT_OVERLAP", "2")),
        int(os.environ.get("DOCUMENT_PAD_GUARD", "0")),
        # Unset means "read it from the checkpoint", which is the intended
        # path; setting it turns into an assertion against the checkpoint.
        (os.environ["SPLADE_FOLD_VOCAB_MASK"].lower() in truthy
         if "SPLADE_FOLD_VOCAB_MASK" in os.environ else None),
    )
    app.state.sem = asyncio.Semaphore(
        int(os.environ.get("BACKEND_MAX_CONCURRENCY", "1"))
    )
    yield


app = FastAPI(title="SPLADE Backend", lifespan=lifespan)


@app.middleware("http")
async def authenticate(request, call_next):
    expected = os.environ.get("BACKEND_API_KEY", "")
    if not expected or request.url.path == "/health":
        return await call_next(request)
    value = request.headers.get("authorization", "")
    if value[:7].lower() == "bearer " and secrets.compare_digest(
        value[7:].strip(), expected
    ):
        return await call_next(request)
    return JSONResponse(status_code=401, content={"detail": "invalid api key"})


@app.get("/health")
async def health():
    return {"ok": True}


@app.get("/metadata")
async def metadata(request: Request):
    data = model_metadata()
    data["device"] = str(request.app.state.encoder.device)
    encoder = request.app.state.encoder
    data["encoder_implementation"] = (
        encoder.encoder_implementation
        if encoder.optimized_document_encoder
        else "reference-pytorch-fp32"
    )
    data["optimized_document_encoder"] = encoder.optimized_document_encoder
    data["document_compute_dtype"] = encoder.document_dtype_name
    # Queries run through the same forward as documents (`SpladeEncoder.encode`
    # serves both), so they come out of the same arithmetic. Reported because
    # this is the question the old constant `"precision": "float32"` answered,
    # and answered wrongly for every profile that casts or autocasts.
    data["query_compute_dtype"] = encoder.document_dtype_name
    data["document_encoding_version"] = encoder.document_encoding_version
    data["document_transports"] = ["json-map-v1", "splade-u16-f16-batch-v1"]
    data["document_weights_cast"] = encoder.weights_cast
    data["document_compile_head"] = encoder.compile_head
    data["document_pad_guard"] = encoder.pad_guard
    data["document_batch_size"] = encoder.batch_size
    data["document_overlap_depth"] = encoder.overlap
    data["document_bit_exact"] = not (encoder.weights_cast or encoder.compile_head)
    # The mask is part of the model contract; expose it in full so a client can
    # pin the exact dimension set instead of trusting two booleans.
    data.update(encoder.mask_detail)
    if encoder.device.type == "cuda":
        data["gpu_name"] = torch.cuda.get_device_name(encoder.device)
        data["gpu_capability"] = list(torch.cuda.get_device_capability(encoder.device))
    return data


@app.post("/encode")
async def encode(body: EncodeRequest, request: Request):
    if not body.inputs:
        raise HTTPException(status_code=400, detail="inputs must not be empty")
    if len(body.inputs) > int(os.environ.get("MAX_INPUTS_PER_REQUEST", "2048")):
        raise HTTPException(status_code=413, detail="too many inputs")
    async with request.app.state.sem:
        return await asyncio.to_thread(
            request.app.state.encoder.encode,
            body.inputs,
            body.document,
        )


@app.post("/encode-packed")
async def encode_packed(body: EncodeRequest, request: Request):
    if not body.document:
        raise HTTPException(status_code=400, detail="packed transport is document-only")
    if not body.inputs:
        raise HTTPException(status_code=400, detail="inputs must not be empty")
    if len(body.inputs) > int(os.environ.get("MAX_INPUTS_PER_REQUEST", "2048")):
        raise HTTPException(status_code=413, detail="too many inputs")
    async with request.app.state.sem:
        value = await asyncio.to_thread(
            request.app.state.encoder.encode_packed,
            body.inputs,
        )
    return Response(
        content=value,
        media_type="application/vnd.simplesystem.splade-u16-f16-batch-v1",
    )
