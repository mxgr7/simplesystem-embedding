import json
import io
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from omegaconf import OmegaConf

from embedding_train.index_build import build_arg_parser, run_index_build


TOKEN_IDS = {
    "bolt": 1,
    "screw": 2,
    "nut": 3,
}


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
            [[TOKEN_IDS.get(text, 9)] for text in texts],
            dtype=torch.long,
        )
        attention_mask = torch.ones_like(input_ids)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }


class _IndexModelStub:
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
        vectors = []
        for value in inputs["input_ids"].squeeze(1).tolist():
            if value == 1:
                vectors.append([1.0, 0.0, 0.0])
            elif value == 2:
                vectors.append([0.0, 1.0, 0.0])
            elif value == 3:
                vectors.append([0.0, 0.0, 1.0])
            else:
                vectors.append([0.5, 0.5, 0.0])

        return torch.tensor(vectors, dtype=torch.float32)


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


class IndexBuildCliTests(unittest.TestCase):
    def write_offer_table(self, path):
        table = pa.Table.from_pylist(
            [
                {"offer_id_b64": "o1", "name": "bolt", "label": "Exact"},
                {"offer_id_b64": "o4", "name": "nut", "label": "Exact"},
                {"offer_id_b64": "o2", "name": "", "label": "Irrelevant"},
                {"offer_id_b64": "o3", "name": "screw", "label": "Exact"},
            ]
        )
        pq.write_table(table, path)

    @patch("embedding_train.infer.AutoTokenizer.from_pretrained")
    @patch("embedding_train.index_build.load_embedding_module_from_checkpoint")
    def test_builds_faiss_index_and_metadata(self, load_checkpoint, from_pretrained):
        load_checkpoint.return_value = (_IndexModelStub(), build_cfg())
        from_pretrained.return_value = _TokenizerStub()

        with TemporaryDirectory() as tmp_dir:
            input_path = Path(tmp_dir) / "offers.parquet"
            output_path = Path(tmp_dir) / "offer-index"
            self.write_offer_table(input_path)
            stdout = io.StringIO()

            args = build_arg_parser().parse_args(
                [
                    "--checkpoint",
                    str(Path(tmp_dir) / "model.ckpt"),
                    "--input",
                    str(input_path),
                    "--output",
                    str(output_path),
                ]
            )

            with redirect_stdout(stdout):
                run_index_build(args)

            metadata_rows = pq.read_table(output_path / "metadata.parquet").to_pylist()
            manifest = json.loads((output_path / "manifest.json").read_text())

            self.assertTrue((output_path / "index.faiss").exists())
            self.assertEqual(len(metadata_rows), 3)
            self.assertEqual([row["faiss_id"] for row in metadata_rows], [0, 1, 2])
            self.assertEqual([row["row_number"] for row in metadata_rows], [0, 1, 3])
            self.assertEqual(
                [row["offer_text"] for row in metadata_rows],
                ["bolt", "nut", "screw"],
            )
            self.assertEqual(
                [row["offer_id_b64"] for row in metadata_rows],
                ["o1", "o4", "o3"],
            )
            self.assertEqual(manifest["embedding_dim"], 3)
            self.assertEqual(manifest["indexed_rows"], 3)
            self.assertEqual(manifest["skipped_rows"], 1)
            self.assertEqual(manifest["index_type"], "flat")
            self.assertIn("Index build complete", stdout.getvalue())

    @patch("embedding_train.infer.AutoTokenizer.from_pretrained")
    @patch("embedding_train.index_build.EmbeddingModule")
    @patch("embedding_train.index_build.load_base_config")
    def test_builds_index_without_checkpoint(
        self,
        load_base_config,
        embedding_module,
        from_pretrained,
    ):
        load_base_config.return_value = build_cfg()
        embedding_module.return_value = _IndexModelStub()
        from_pretrained.return_value = _TokenizerStub()

        with TemporaryDirectory() as tmp_dir:
            input_path = Path(tmp_dir) / "offers.parquet"
            output_path = Path(tmp_dir) / "offer-index"
            self.write_offer_table(input_path)

            args = build_arg_parser().parse_args(
                [
                    "--input",
                    str(input_path),
                    "--output",
                    str(output_path),
                    "--model-name",
                    "custom-stub-model",
                ]
            )

            run_index_build(args)
            manifest = json.loads((output_path / "manifest.json").read_text())

        self.assertEqual(manifest["checkpoint"], "")
        self.assertEqual(manifest["query_model_name"], "custom-stub-model")
        embedding_module.assert_called_once()

    @patch("embedding_train.infer.AutoTokenizer.from_pretrained")
    @patch("embedding_train.index_build.load_embedding_module_from_checkpoint")
    def test_persists_ann_index_configuration(self, load_checkpoint, from_pretrained):
        load_checkpoint.return_value = (_IndexModelStub(), build_cfg())
        from_pretrained.return_value = _TokenizerStub()

        with TemporaryDirectory() as tmp_dir:
            input_path = Path(tmp_dir) / "offers.parquet"
            output_path = Path(tmp_dir) / "offer-index"
            self.write_offer_table(input_path)

            args = build_arg_parser().parse_args(
                [
                    "--checkpoint",
                    str(Path(tmp_dir) / "model.ckpt"),
                    "--input",
                    str(input_path),
                    "--output",
                    str(output_path),
                    "--index-type",
                    "ivf_pq",
                    "--nlist",
                    "1",
                    "--train-sample-size",
                    "3",
                    "--pq-m",
                    "1",
                    "--pq-bits",
                    "1",
                    "--nprobe",
                    "1",
                ]
            )

            run_index_build(args)
            manifest = json.loads((output_path / "manifest.json").read_text())

        self.assertEqual(manifest["index_type"], "ivf_pq")
        self.assertEqual(manifest["index_config"]["nlist"], 1)
        self.assertEqual(manifest["index_config"]["pq_m"], 1)
        self.assertEqual(manifest["index_config"]["pq_bits"], 1)
        self.assertEqual(manifest["index_config"]["nprobe"], 1)


if __name__ == "__main__":
    unittest.main()
