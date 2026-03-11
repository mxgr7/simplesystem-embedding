import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from omegaconf import OmegaConf

from embedding_train.infer import build_arg_parser, run_inference


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


class _InferenceModelStub:
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
            },
        }
    )


class InferenceCliTests(unittest.TestCase):
    def write_input_table(self, path):
        table = pa.Table.from_pylist(
            [
                {
                    "query_id": "q1",
                    "offer_id_b64": "o1",
                    "query_term": "nut",
                    "name": "hex nut",
                    "label": "Exact",
                },
                {
                    "query_id": "q2",
                    "offer_id_b64": "o2",
                    "query_term": "bolt",
                    "name": "",
                    "label": "Irrelevant",
                },
                {
                    "query_id": "q3",
                    "offer_id_b64": "o3",
                    "query_term": "screw",
                    "name": "wood screw",
                    "label": "Exact",
                },
            ]
        )
        pq.write_table(table, path)

    @patch("embedding_train.infer.AutoTokenizer.from_pretrained")
    @patch("embedding_train.infer.load_embedding_module_from_checkpoint")
    def test_exports_embeddings_in_chunked_batches(
        self, load_checkpoint, from_pretrained
    ):
        load_checkpoint.return_value = (_InferenceModelStub(), build_cfg())
        from_pretrained.return_value = _TokenizerStub()

        with TemporaryDirectory() as tmp_dir:
            input_path = Path(tmp_dir) / "input.parquet"
            output_path = Path(tmp_dir) / "output.parquet"
            self.write_input_table(input_path)

            args = build_arg_parser().parse_args(
                [
                    "--checkpoint",
                    str(Path(tmp_dir) / "model.ckpt"),
                    "--input",
                    str(input_path),
                    "--output",
                    str(output_path),
                    "--mode",
                    "offer",
                    "--read-batch-size",
                    "2",
                    "--encode-batch-size",
                    "1",
                    "--include-text",
                ]
            )

            run_inference(args)
            output_rows = pq.read_table(output_path).to_pylist()

        self.assertEqual(len(output_rows), 2)
        self.assertEqual([row["row_number"] for row in output_rows], [0, 2])
        self.assertEqual([row["query_id"] for row in output_rows], ["q1", "q3"])
        self.assertEqual(output_rows[0]["offer_text"], "hex nut")
        self.assertEqual(output_rows[0]["offer_embedding"], [7.0, 8.0])
        self.assertEqual(output_rows[1]["offer_embedding"], [10.0, 11.0])

    @patch("embedding_train.infer.AutoTokenizer.from_pretrained")
    @patch("embedding_train.infer.load_embedding_module_from_checkpoint")
    def test_pair_score_mode_scores_paired_rows(self, load_checkpoint, from_pretrained):
        load_checkpoint.return_value = (_InferenceModelStub(), build_cfg())
        from_pretrained.return_value = _TokenizerStub()

        with TemporaryDirectory() as tmp_dir:
            input_path = Path(tmp_dir) / "input.parquet"
            output_path = Path(tmp_dir) / "scores.parquet"
            self.write_input_table(input_path)

            args = build_arg_parser().parse_args(
                [
                    "--checkpoint",
                    str(Path(tmp_dir) / "model.ckpt"),
                    "--input",
                    str(input_path),
                    "--output",
                    str(output_path),
                    "--mode",
                    "pair_score",
                    "--read-batch-size",
                    "2",
                    "--encode-batch-size",
                    "1",
                ]
            )

            run_inference(args)
            output_rows = pq.read_table(output_path).to_pylist()

        self.assertEqual(len(output_rows), 2)
        self.assertEqual([row["pair_score"] for row in output_rows], [53.0, 116.0])


if __name__ == "__main__":
    unittest.main()
