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

from embedding_train.index_build import build_arg_parser as build_index_arg_parser
from embedding_train.index_build import run_index_build
from embedding_train.index_search import build_arg_parser, main, run_index_search


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


class IndexSearchCliTests(unittest.TestCase):
    def write_offer_table(self, path):
        table = pa.Table.from_pylist(
            [
                {"offer_id_b64": "o1", "name": "bolt", "label": "Exact"},
                {"offer_id_b64": "o2", "name": "nut", "label": "Exact"},
                {"offer_id_b64": "o3", "name": "screw", "label": "Exact"},
            ]
        )
        pq.write_table(table, path)

    def write_query_table(self, path):
        table = pa.Table.from_pylist(
            [
                {"query_id": "q1", "query_term": "bolt", "label": "Exact"},
                {"query_id": "q2", "query_term": "screw", "label": "Exact"},
                {"query_id": "q3", "query_term": "", "label": "Exact"},
            ]
        )
        pq.write_table(table, path)

    def build_index(self, tmp_dir, extra_args=None):
        input_path = Path(tmp_dir) / "offers.parquet"
        index_path = Path(tmp_dir) / "offer-index"
        self.write_offer_table(input_path)
        cli_args = [
            "--checkpoint",
            str(Path(tmp_dir) / "model.ckpt"),
            "--input",
            str(input_path),
            "--output",
            str(index_path),
        ]
        if extra_args:
            cli_args.extend(extra_args)

        args = build_index_arg_parser().parse_args(cli_args)
        run_index_build(args)
        return index_path

    @patch("embedding_train.infer.AutoTokenizer.from_pretrained")
    @patch("embedding_train.index_search.load_embedding_module_from_checkpoint")
    @patch("embedding_train.index_build.load_embedding_module_from_checkpoint")
    def test_searches_built_index_from_parquet_queries(
        self,
        build_load_checkpoint,
        search_load_checkpoint,
        from_pretrained,
    ):
        build_load_checkpoint.return_value = (_IndexModelStub(), build_cfg())
        search_load_checkpoint.return_value = (_IndexModelStub(), build_cfg())
        from_pretrained.return_value = _TokenizerStub()

        with TemporaryDirectory() as tmp_dir:
            index_path = self.build_index(tmp_dir)
            query_path = Path(tmp_dir) / "queries.parquet"
            output_path = Path(tmp_dir) / "results.parquet"
            self.write_query_table(query_path)

            args = build_arg_parser().parse_args(
                [
                    "--checkpoint",
                    str(Path(tmp_dir) / "model.ckpt"),
                    "--index",
                    str(index_path),
                    "--input",
                    str(query_path),
                    "--output",
                    str(output_path),
                    "--top-k",
                    "2",
                ]
            )

            run_index_search(args)
            result_rows = pq.read_table(output_path).to_pylist()

        self.assertEqual(len(result_rows), 4)
        self.assertEqual(result_rows[0]["query_id"], "q1")
        self.assertEqual(result_rows[0]["query_text"], "bolt")
        self.assertEqual(result_rows[0]["rank"], 1)
        self.assertEqual(result_rows[0]["match_offer_id_b64"], "o1")
        self.assertEqual(result_rows[1]["rank"], 2)
        self.assertEqual(result_rows[2]["query_id"], "q2")
        self.assertEqual(result_rows[2]["match_offer_id_b64"], "o3")
        self.assertEqual(result_rows[3]["rank"], 2)

    @patch("embedding_train.infer.AutoTokenizer.from_pretrained")
    @patch("embedding_train.index_search.load_embedding_module_from_checkpoint")
    @patch("embedding_train.index_build.load_embedding_module_from_checkpoint")
    def test_searches_built_index_from_raw_query_text_and_prints_report(
        self,
        build_load_checkpoint,
        search_load_checkpoint,
        from_pretrained,
    ):
        build_load_checkpoint.return_value = (_IndexModelStub(), build_cfg())
        search_load_checkpoint.return_value = (_IndexModelStub(), build_cfg())
        from_pretrained.return_value = _TokenizerStub()

        with TemporaryDirectory() as tmp_dir:
            index_path = self.build_index(tmp_dir)
            stdout = io.StringIO()

            with redirect_stdout(stdout):
                main(
                    [
                        "--checkpoint",
                        str(Path(tmp_dir) / "model.ckpt"),
                        "--index",
                        str(index_path),
                        "--query-text",
                        "bolt",
                        "--top-k",
                        "1",
                    ]
                )

        output = stdout.getvalue()
        self.assertIn("FAISS Search Results", output)
        self.assertIn("bolt", output)
        self.assertIn("o1", output)

    @patch("embedding_train.infer.AutoTokenizer.from_pretrained")
    @patch("embedding_train.index_search.load_embedding_module_from_checkpoint")
    @patch("embedding_train.index_build.load_embedding_module_from_checkpoint")
    def test_search_supports_ann_index_types(
        self,
        build_load_checkpoint,
        search_load_checkpoint,
        from_pretrained,
    ):
        build_load_checkpoint.return_value = (_IndexModelStub(), build_cfg())
        search_load_checkpoint.return_value = (_IndexModelStub(), build_cfg())
        from_pretrained.return_value = _TokenizerStub()

        cases = [
            (
                "ivf_flat",
                [
                    "--index-type",
                    "ivf_flat",
                    "--nlist",
                    "1",
                    "--train-sample-size",
                    "3",
                    "--nprobe",
                    "1",
                ],
                ["--nprobe", "1"],
            ),
            (
                "ivf_pq",
                [
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
                ],
                ["--nprobe", "1"],
            ),
            (
                "hnsw",
                [
                    "--index-type",
                    "hnsw",
                    "--hnsw-m",
                    "4",
                    "--ef-search",
                    "8",
                    "--ef-construction",
                    "8",
                ],
                ["--ef-search", "8"],
            ),
        ]

        for index_type, build_args, search_args in cases:
            with self.subTest(index_type=index_type), TemporaryDirectory() as tmp_dir:
                index_path = self.build_index(tmp_dir, extra_args=build_args)

                cli_args = [
                    "--checkpoint",
                    str(Path(tmp_dir) / "model.ckpt"),
                    "--index",
                    str(index_path),
                    "--query-text",
                    "bolt",
                    "--top-k",
                    "1",
                ]
                cli_args.extend(search_args)

                args = build_arg_parser().parse_args(cli_args)

                result_rows = run_index_search(args)

                self.assertEqual(len(result_rows), 1)
                self.assertEqual(result_rows[0]["match_offer_id_b64"], "o1")


if __name__ == "__main__":
    unittest.main()
