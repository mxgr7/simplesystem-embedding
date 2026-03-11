import math
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from omegaconf import OmegaConf

from embedding_train.eval import build_arg_parser, run_evaluation


class _TokenizerStub:
    pad_token = "[PAD]"
    eos_token = "[EOS]"

    def __call__(
        self,
        texts,
        padding,
        truncation,
        max_length,
        return_tensors,
    ):
        del padding, truncation, max_length, return_tensors
        input_ids = torch.tensor(
            [[max(1, len(text))] for text in texts], dtype=torch.long
        )
        attention_mask = torch.ones_like(input_ids)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }


class _EvaluationModelStub:
    def __init__(self):
        self.device = torch.device("cpu")
        self.training = True

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        self.training = False
        return self

    def encode(self, inputs):
        values = inputs["input_ids"].squeeze(1).float()
        return torch.stack([values, values + 1], dim=1)


def build_cfg():
    return OmegaConf.create(
        {
            "model": {"model_name": "stub-model", "output_dim": None},
            "data": {
                "query_template": "{{ query_term }}",
                "offer_template": "{{ name }}",
                "clean_html": True,
                "max_query_length": 32,
                "max_offer_length": 64,
                "positive_label": "Exact",
            },
        }
    )


class EvaluationCliTests(unittest.TestCase):
    def write_input_table(self, path):
        table = pa.Table.from_pylist(
            [
                {
                    "query_id": "q1",
                    "offer_id_b64": "o1",
                    "query_term": "nut",
                    "name": "a",
                    "label": "Irrelevant",
                },
                {
                    "query_id": "q1",
                    "offer_id_b64": "o2",
                    "query_term": "nut",
                    "name": "long exact",
                    "label": "Exact",
                },
            ]
        )
        pq.write_table(table, path)

    @patch("embedding_train.infer.AutoTokenizer.from_pretrained")
    @patch("embedding_train.eval.load_embedding_module_from_checkpoint")
    def test_reports_ndcg_delta_for_binary_precision(
        self, load_checkpoint, from_pretrained
    ):
        load_checkpoint.return_value = (_EvaluationModelStub(), build_cfg())
        from_pretrained.return_value = _TokenizerStub()

        with TemporaryDirectory() as tmp_dir:
            input_path = Path(tmp_dir) / "input.parquet"
            self.write_input_table(input_path)

            args = build_arg_parser().parse_args(
                [
                    "--checkpoint",
                    str(Path(tmp_dir) / "model.ckpt"),
                    "--input",
                    str(input_path),
                    "--embedding-precision",
                    "binary",
                ]
            )

            report = run_evaluation(args)

        self.assertEqual(report["embedding_precision"], "binary")
        self.assertEqual(report["baseline_precision"], "float32")
        self.assertEqual(report["processed_rows"], 2.0)
        self.assertEqual(report["evaluated_rows"], 2.0)
        self.assertEqual(report["skipped_rows"], 0.0)
        self.assertTrue(math.isclose(report["metrics"]["ndcg@1"], 0.0))
        self.assertTrue(math.isclose(report["metrics"]["ndcg@5"], 0.6309297535714575))
        self.assertTrue(math.isclose(report["baseline_metrics"]["ndcg@1"], 1.0))
        self.assertTrue(math.isclose(report["metric_deltas"]["ndcg@1"], -1.0))


if __name__ == "__main__":
    unittest.main()
