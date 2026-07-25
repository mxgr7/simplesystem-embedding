import gc
import os
import sys
from pathlib import Path

import pytest
import torch


REPO = Path(__file__).resolve().parents[1]
SERVICE = REPO / "splade-service"
sys.path.insert(0, str(SERVICE))
sys.path.insert(0, str(REPO / "src"))

CHECKPOINT = os.environ.get("SPLADE_PARITY_CHECKPOINT")
pytestmark = pytest.mark.skipif(
    not CHECKPOINT,
    reason="set SPLADE_PARITY_CHECKPOINT to run the production model parity test",
)


def test_standalone_backend_matches_training_model_exactly(tmp_path):
    from backend import SpladeEncoder
    from constants import FIELD_ORDER, TOP_K
    from rendering import render_from_nul

    row = {name: "" for name in FIELD_ORDER}
    row.update({
        "name": "Kühlschrank Pro",
        "manufacturer_name": "Müller",
        "ean": "4000123456789",
        "features_text": "Größe: 60 cm.",
    })
    text = render_from_nul("\x00".join(row[name] for name in FIELD_ORDER))

    standalone = SpladeEncoder(CHECKPOINT, "cpu", str(tmp_path), 1)
    actual = standalone.encode([text])[0]
    del standalone
    gc.collect()

    from embedding_train.model import load_embedding_module_from_checkpoint
    from embedding_train.tokenization import load_fast_tokenizer

    reference, config = load_embedding_module_from_checkpoint(CHECKPOINT)
    tokenizer = load_fast_tokenizer(config.model.model_name)
    tokens = tokenizer([text], padding=True, return_tensors="pt")
    with torch.inference_mode():
        vector = reference.encode(tokens).float()[0]
    values, ids = torch.topk(vector, TOP_K, sorted=True)
    expected = {
        str(int(token_id)): float(weight)
        for token_id, weight in zip(ids, values)
        if weight > 0
    }
    assert actual == expected
