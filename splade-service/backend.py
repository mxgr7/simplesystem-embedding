import asyncio
import hashlib
import json
import os
import secrets
from contextlib import asynccontextmanager

import torch
import transformers
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
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


class EncodeRequest(BaseModel):
    inputs: list[str]
    document: bool = True


def checkpoint_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_tokenizer(model_name, cache_dir):
    try:
        return AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    except (ValueError, OSError):
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


class SpladeEncoder:
    def __init__(self, checkpoint_path, device, cache_dir, batch_size):
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
        self.model = model.to(device=self.device, dtype=torch.float32).eval()
        self.tokenizer = load_tokenizer(MODEL_NAME, cache_dir)
        self.mask = torch.ones(config.vocab_size, dtype=torch.float32)
        self.mask[self.tokenizer.all_special_ids] = 0
        self.mask = self.mask.to(self.device)
        self.batch_size = batch_size

    def encode(self, texts, document=True):
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
            with torch.inference_mode():
                logits = self.model(
                    input_ids=tokens["input_ids"],
                    attention_mask=tokens["attention_mask"],
                ).logits
                activations = torch.log1p(torch.relu(logits))
                activations *= tokens["attention_mask"].unsqueeze(-1).to(
                    activations.dtype
                )
                vectors = activations.amax(dim=1) * self.mask

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
            del logits, activations, vectors
        return output


@asynccontextmanager
async def lifespan(app):
    checkpoint = os.environ.get("SPLADE_CHECKPOINT", "/model/prod_soup.ckpt")
    device = os.environ.get("DEVICE", "auto")
    cache_dir = os.environ.get("HF_HOME", "/data/huggingface")
    batch_size = int(os.environ.get("BACKEND_BATCH_SIZE", "2"))
    threads = int(os.environ.get("TORCH_NUM_THREADS", str(os.cpu_count() or 1)))
    torch.set_num_threads(threads)
    app.state.encoder = await asyncio.to_thread(
        SpladeEncoder, checkpoint, device, cache_dir, batch_size
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
    return data


@app.post("/encode")
async def encode(body: EncodeRequest, request: Request):
    if not body.inputs:
        raise HTTPException(status_code=400, detail="inputs must not be empty")
    if len(body.inputs) > int(os.environ.get("MAX_INPUTS_PER_REQUEST", "256")):
        raise HTTPException(status_code=413, detail="too many inputs")
    async with request.app.state.sem:
        return await asyncio.to_thread(
            request.app.state.encoder.encode,
            body.inputs,
            body.document,
        )
