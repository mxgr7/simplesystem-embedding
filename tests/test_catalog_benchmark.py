import io
import math
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from omegaconf import OmegaConf

from embedding_train.catalog_benchmark import (
    build_arg_parser,
    main,
    run_catalog_benchmark,
)


TOKEN_IDS = {
    "query-a": 1,
    "query-b": 2,
    "query-c": 3,
    "sub-a": 4,
    "exact-a": 5,
    "exact-b": 6,
    "sub-only": 7,
    "irrelevant": 8,
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
            [[TOKEN_IDS.get(text, 99)] for text in texts],
            dtype=torch.long,
        )
        attention_mask = torch.ones_like(input_ids)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }


class _BenchmarkModelStub:
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
            elif value == 4:
                vectors.append([1.0, 0.0, 0.0])
            elif value == 5:
                vectors.append([0.9, 0.1, 0.0])
            elif value == 6:
                vectors.append([0.0, 1.0, 0.0])
            elif value == 7:
                vectors.append([0.0, 0.0, 1.0])
            else:
                vectors.append([0.1, 0.1, 0.1])

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
                "column_mapping": {
                    "query_id": "query_id",
                    "offer_id": "offer_id_b64",
                },
            },
        }
    )


class CatalogBenchmarkCliTests(unittest.TestCase):
    def write_input_table(self, path):
        table = pa.Table.from_pylist(
            [
                {
                    "query_id": "q1",
                    "query_term": "query-a",
                    "offer_id_b64": "o1",
                    "name": "sub-a",
                    "label": "Substitute",
                },
                {
                    "query_id": "q1",
                    "query_term": "query-a",
                    "offer_id_b64": "o2",
                    "name": "exact-a",
                    "label": "Complement",
                },
                {
                    "query_id": "q1",
                    "query_term": "query-a",
                    "offer_id_b64": "o2",
                    "name": "exact-a",
                    "label": "Exact",
                },
                {
                    "query_id": "q2",
                    "query_term": "query-b",
                    "offer_id_b64": "o3",
                    "name": "exact-b",
                    "label": "Exact",
                },
                {
                    "query_id": "q3",
                    "query_term": "query-c",
                    "offer_id_b64": "o4",
                    "name": "sub-only",
                    "label": "Substitute",
                },
                {
                    "query_id": "q3",
                    "query_term": "query-c",
                    "offer_id_b64": "o5",
                    "name": "irrelevant",
                    "label": "Irrelevant",
                },
                {
                    "query_id": "",
                    "query_term": "query-b",
                    "offer_id_b64": "o3",
                    "name": "exact-b",
                    "label": "Exact",
                },
                {
                    "query_id": "q4",
                    "query_term": "",
                    "offer_id_b64": "o6",
                    "name": "",
                    "label": "Exact",
                },
            ]
        )
        pq.write_table(table, path)

    @patch("embedding_train.infer.AutoTokenizer.from_pretrained")
    @patch("embedding_train.catalog_benchmark.load_embedding_module_from_checkpoint")
    def test_runs_exact_catalog_benchmark_with_expected_metrics(
        self, load_checkpoint, from_pretrained
    ):
        load_checkpoint.return_value = (_BenchmarkModelStub(), build_cfg())
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
                    "--ks",
                    "1,2",
                    "--score-batch-size",
                    "2",
                ]
            )

            report = run_catalog_benchmark(args)

        q1_ndcg_at_2 = (0.1 + (1.0 / math.log2(3.0))) / (1.0 + (0.1 / math.log2(3.0)))
        expected_ndcg_at_2 = (q1_ndcg_at_2 + 1.0 + 1.0) / 3.0

        self.assertEqual(report["similarity"], "dot")
        self.assertEqual(report["ks"], (1, 2))
        self.assertEqual(report["relevant_labels"], ("Exact",))
        self.assertEqual(report["processed_rows"], 8.0)
        self.assertEqual(report["skipped_rows"], 2.0)
        self.assertEqual(report["query_count"], 3.0)
        self.assertEqual(report["catalog_size"], 5.0)
        self.assertEqual(report["metrics"]["evaluated_queries"], 3.0)
        self.assertEqual(report["metrics"]["ndcg_eligible_queries"], 3.0)
        self.assertEqual(report["metrics"]["retrieval_eligible_queries"], 2.0)
        self.assertTrue(math.isclose(report["metrics"]["mrr"], 0.75))
        self.assertTrue(math.isclose(report["metrics"]["ndcg@1"], 0.7))
        self.assertTrue(math.isclose(report["metrics"]["ndcg@2"], expected_ndcg_at_2))
        self.assertTrue(math.isclose(report["metrics"]["recall@1"], 0.5))
        self.assertTrue(math.isclose(report["metrics"]["recall@2"], 1.0))
        self.assertTrue(math.isclose(report["metrics"]["precision@1"], 0.5))
        self.assertTrue(math.isclose(report["metrics"]["precision@2"], 0.5))

    @patch("embedding_train.infer.AutoTokenizer.from_pretrained")
    @patch("embedding_train.catalog_benchmark.load_embedding_module_from_checkpoint")
    def test_main_prints_formatted_report(self, load_checkpoint, from_pretrained):
        load_checkpoint.return_value = (_BenchmarkModelStub(), build_cfg())
        from_pretrained.return_value = _TokenizerStub()

        with TemporaryDirectory() as tmp_dir:
            input_path = Path(tmp_dir) / "input.parquet"
            self.write_input_table(input_path)

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                main(
                    [
                        "--checkpoint",
                        str(Path(tmp_dir) / "model.ckpt"),
                        "--input",
                        str(input_path),
                        "--ks",
                        "1,2",
                    ]
                )

        output = stdout.getvalue()
        self.assertIn("Catalog Benchmark", output)
        self.assertIn("Similarity", output)
        self.assertIn("Metrics", output)
        self.assertIn("ndcg@1", output)
        self.assertIn("recall@2", output)
        self.assertIn("precision@2", output)


if __name__ == "__main__":
    unittest.main()
