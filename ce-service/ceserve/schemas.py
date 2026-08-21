"""Pydantic models for OpenAPI ONLY — the handler never validates through them.

`cross_encoder_serve` measured ~100 ms of per-item pydantic validation on a
2,000-offer request. At the 120-candidate window this service is sized for that
is ~6 ms of a 150 ms budget given away for nothing, and it scales linearly
toward the 256 cap. The `/rerank` handler therefore parses the raw body with
orjson and validates by hand (`app._parse_request`), while these models stay on
the route decorator so `/docs` still describes the contract.

MXG-144.
"""
from pydantic import BaseModel, Field


class RerankCandidate(BaseModel):
    id: str = Field(description="Opaque, echoed back verbatim. Must be unique.")
    tokens_b64: str = Field(
        description="The ES `binary` field `ceTokenIds`, VERBATIM as the base64 "
        "string `_source` returned. Decodes to little-endian int32 per id, no "
        "special tokens. The caller never decodes it."
    )
    token_count: int = Field(
        description="The ES `ceTokenCount`. Redundant with len(tokens_b64); "
        "asserted, and the assertion is the point — a disagreement means a "
        "truncated or rewritten _source, which is otherwise invisible."
    )
    tokenizer_version: str = Field(
        description="The ES `ceTokenizerVersion`, PER CANDIDATE. A partially "
        "backfilled index carries a mix, and surviving that is why the field "
        "exists. A mismatch lands the candidate in `skipped`, never in `results`."
    )


class RerankRequest(BaseModel):
    query: str = Field(description="The UNTOUCHED raw user query. This service "
                                   "alone builds the model input: `fold_de` "
                                   "(German umlaut expansion, casefold, strip "
                                   "remaining combining marks; whitespace "
                                   "preserved) and NO prefix — query contract "
                                   "fold-de-v1-no-prefix. A request carrying a "
                                   "`segment` key, null included, is a 400.")
    max_len: int | None = Field(
        default=None,
        description="Serve-time width dial, clamped to [8, CE_MAX_LEN]. Quality "
        "is flat 128->256, so this trades window depth against width with no "
        "redeploy.",
    )
    debug: bool = Field(default=False, description="Adds a `timings` object.")
    candidates: list[RerankCandidate]


class ScoredCandidate(BaseModel):
    id: str
    ce_score: float = Field(description="sum(softmax(logits) * [4,2,1,0]) / 4 — "
                                        "`train_ce.py`'s ce_score.")
    ce_p_e: float
    ce_p_s: float
    ce_p_c: float
    ce_p_i: float
    article_token_count: int
    truncated: bool = Field(description="True if the stored prefix was longer "
                                        "than the splice budget.")


class SkippedCandidate(BaseModel):
    id: str
    reason: str = Field(description="tokenizer_version_mismatch | decode_failed | "
                                    "no_tokens")
    detail: str


class RerankResponse(BaseModel):
    model_id: str
    model_sha256: str
    tokenizer_version: str
    serving_contract: str
    query_contract: str = Field(description="fold-de-v1-no-prefix. Callers "
                                            "assert on this alongside "
                                            "serving_contract.")
    declined_reason: str | None = Field(
        default=None,
        description="Present (as `empty_folded_query`) only when a nonblank "
        "raw query folds to nothing encodable: HTTP 200, no inference, and "
        "BOTH arrays empty — the one carve-out to the every-id-comes-back "
        "contract.",
    )
    max_len: int
    n_input: int
    n_scored: int
    n_skipped: int
    query_token_count: int
    padded_width: int
    results: list[ScoredCandidate] = Field(
        description="IN REQUEST ORDER, not sorted by score. This service does "
        "not own the ranking policy — EAN pin-to-top, tail demotion and tie "
        "handling all live in the query service's RerankerService."
    )
    skipped: list[SkippedCandidate]
