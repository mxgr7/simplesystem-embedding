import io
import math
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from omegaconf import OmegaConf

from embedding_train.index_build import build_arg_parser as build_index_arg_parser
from embedding_train.index_build import run_index_build
from embedding_train.eval import build_arg_parser, main, run_evaluation


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
            [[TOKEN_IDS.get(text, max(1, len(text)))] for text in texts],
            dtype=torch.long,
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


class _RetrievalModelStub:
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

    def write_offer_table(self, path):
        table = pa.Table.from_pylist(
            [
                {"offer_id_b64": "o1", "name": "bolt", "label": "Exact"},
                {"offer_id_b64": "o2", "name": "nut", "label": "Exact"},
                {"offer_id_b64": "o3", "name": "screw", "label": "Exact"},
            ]
        )
        pq.write_table(table, path)

    def write_retrieval_input_table(self, path):
        table = pa.Table.from_pylist(
            [
                {
                    "query_id": "q1",
                    "offer_id_b64": "o1",
                    "query_term": "bolt",
                    "name": "bolt",
                    "label": "Exact",
                },
                {
                    "query_id": "q1",
                    "offer_id_b64": "o2",
                    "query_term": "bolt",
                    "name": "nut",
                    "label": "Irrelevant",
                },
                {
                    "query_id": "q2",
                    "offer_id_b64": "o3",
                    "query_term": "screw",
                    "name": "screw",
                    "label": "Exact",
                },
                {
                    "query_id": "q2",
                    "offer_id_b64": "o1",
                    "query_term": "screw",
                    "name": "bolt",
                    "label": "Irrelevant",
                },
                {
                    "query_id": "q3",
                    "offer_id_b64": "o2",
                    "query_term": "nut",
                    "name": "nut",
                    "label": "Substitute",
                },
            ]
        )
        pq.write_table(table, path)

    def build_index(self, tmp_dir):
        input_path = Path(tmp_dir) / "offers.parquet"
        output_path = Path(tmp_dir) / "offer-index"
        self.write_offer_table(input_path)

        args = build_index_arg_parser().parse_args(
            [
                "--checkpoint",
                str(Path(tmp_dir) / "model.ckpt"),
                "--input",
                str(input_path),
                "--output",
                str(output_path),
            ]
        )

        run_index_build(args)
        return output_path

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

            stderr = io.StringIO()
            with redirect_stderr(stderr):
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
        self.assertIn("Evaluating", stderr.getvalue())

    @patch("embedding_train.infer.AutoTokenizer.from_pretrained")
    @patch("embedding_train.eval.load_embedding_module_from_checkpoint")
    def test_main_prints_formatted_report(self, load_checkpoint, from_pretrained):
        load_checkpoint.return_value = (_EvaluationModelStub(), build_cfg())
        from_pretrained.return_value = _TokenizerStub()

        with TemporaryDirectory() as tmp_dir:
            input_path = Path(tmp_dir) / "input.parquet"
            self.write_input_table(input_path)

            stdout = io.StringIO()
            stderr = io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                main(
                    [
                        "--checkpoint",
                        str(Path(tmp_dir) / "model.ckpt"),
                        "--input",
                        str(input_path),
                        "--embedding-precision",
                        "binary",
                    ]
                )

        output = stdout.getvalue()
        self.assertIn("Embedding Evaluation", output)
        self.assertIn("Precision       binary", output)
        self.assertIn("Metrics", output)
        self.assertIn("Selected", output)
        self.assertIn("Baseline", output)
        self.assertIn("Delta", output)

    @patch("embedding_train.infer.AutoTokenizer.from_pretrained")
    @patch("embedding_train.eval.load_embedding_module_from_checkpoint")
    @patch("embedding_train.index_build.load_embedding_module_from_checkpoint")
    def test_reports_exact_retrieval_metrics_for_index_search(
        self, build_load_checkpoint, eval_load_checkpoint, from_pretrained
    ):
        build_load_checkpoint.return_value = (_RetrievalModelStub(), build_cfg())
        eval_load_checkpoint.return_value = (_RetrievalModelStub(), build_cfg())
        from_pretrained.return_value = _TokenizerStub()

        with TemporaryDirectory() as tmp_dir:
            index_path = self.build_index(tmp_dir)
            input_path = Path(tmp_dir) / "retrieval_input.parquet"
            self.write_retrieval_input_table(input_path)

            stderr = io.StringIO()
            with redirect_stderr(stderr):
                report = run_evaluation(
                    build_arg_parser().parse_args(
                        [
                            "--checkpoint",
                            str(Path(tmp_dir) / "model.ckpt"),
                            "--input",
                            str(input_path),
                            "--index",
                            str(index_path),
                            "--top-k",
                            "3",
                        ]
                    )
                )

        self.assertEqual(report["evaluation_mode"], "retrieval")
        self.assertEqual(report["index_type"], "flat")
        self.assertEqual(report["processed_rows"], 5.0)
        self.assertEqual(report["searched_queries"], 3.0)
        self.assertEqual(report["skipped_rows"], 0.0)
        self.assertEqual(report["metrics"]["evaluated_queries"], 3.0)
        self.assertEqual(report["metrics"]["eligible_queries"], 2.0)
        self.assertTrue(math.isclose(report["metrics"]["exact_success@1"], 1.0))
        self.assertTrue(math.isclose(report["metrics"]["exact_mrr"], 1.0))
        self.assertIn("Evaluating", stderr.getvalue())

    @patch("embedding_train.infer.AutoTokenizer.from_pretrained")
    @patch("embedding_train.eval.load_embedding_module_from_checkpoint")
    @patch("embedding_train.index_build.load_embedding_module_from_checkpoint")
    def test_main_prints_formatted_retrieval_report(
        self, build_load_checkpoint, eval_load_checkpoint, from_pretrained
    ):
        build_load_checkpoint.return_value = (_RetrievalModelStub(), build_cfg())
        eval_load_checkpoint.return_value = (_RetrievalModelStub(), build_cfg())
        from_pretrained.return_value = _TokenizerStub()

        with TemporaryDirectory() as tmp_dir:
            index_path = self.build_index(tmp_dir)
            input_path = Path(tmp_dir) / "retrieval_input.parquet"
            self.write_retrieval_input_table(input_path)

            stdout = io.StringIO()
            stderr = io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                main(
                    [
                        "--checkpoint",
                        str(Path(tmp_dir) / "model.ckpt"),
                        "--input",
                        str(input_path),
                        "--index",
                        str(index_path),
                        "--top-k",
                        "3",
                    ]
                )

        output = stdout.getvalue()
        self.assertIn("Mode              retrieval", output)
        self.assertIn("exact_success@1", output)
        self.assertIn("exact_mrr", output)


if __name__ == "__main__":
    unittest.main()
