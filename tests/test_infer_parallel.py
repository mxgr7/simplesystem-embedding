import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq

from embedding_train.infer_parallel import (
    build_arg_parser,
    run_parallel,
    split_input_parquet,
)
from tests.test_infer import _InferenceModelStub, _TokenizerStub, build_cfg


class InferParallelTests(unittest.TestCase):
    def write_input_dir(self, path: Path, rows):
        path.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pylist(rows)
        pq.write_table(table, path / "data_0.parquet")

    def make_rows(self, count):
        return [
            {
                "query_id": f"q{i}",
                "offer_id_b64": f"o{i}",
                "query_term": f"term{i}",
                "name": f"item {i}",
                "label": "Exact",
            }
            for i in range(count)
        ]

    def test_split_input_parquet_distributes_rows(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source_dir = tmp_path / "source"
            self.write_input_dir(source_dir, self.make_rows(10))

            shard_paths = [tmp_path / f"s{i}.parquet" for i in range(3)]
            counts = split_input_parquet(source_dir, shard_paths)
            self.assertEqual(sum(counts), 10)
            total = 0
            for path in shard_paths:
                total += pq.ParquetFile(path).metadata.num_rows
            self.assertEqual(total, 10)

    @patch("embedding_train.infer.AutoTokenizer.from_pretrained")
    @patch("embedding_train.infer.load_embedding_module_from_checkpoint")
    def test_end_to_end_sharded_infer(
        self,
        load_checkpoint,
        from_pretrained,
    ):
        load_checkpoint.return_value = (_InferenceModelStub(), build_cfg())
        from_pretrained.return_value = _TokenizerStub()

        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source_dir = tmp_path / "source"
            self.write_input_dir(source_dir, self.make_rows(12))

            output_root = tmp_path / "out"
            args = build_arg_parser().parse_args(
                [
                    "--checkpoint",
                    str(tmp_path / "ckpt"),
                    "--input",
                    str(source_dir),
                    "--output",
                    str(output_root),
                    "--num-shards",
                    "2",
                    "--gpu-ids",
                    "0,0",
                    "--mode",
                    "offer",
                    "--embedding-precision",
                    "float32",
                    "--read-batch-size",
                    "6",
                    "--encode-batch-size",
                    "6",
                    "--compression",
                    "zstd",
                    "--overwrite",
                ]
            )

            with patch(
                "embedding_train.infer_parallel.subprocess.Popen"
            ) as popen_mock:
                # Run the infer command in-process rather than via subprocess
                # so we can patch model/tokenizer loading.
                from embedding_train.infer import (
                    build_arg_parser as infer_arg_parser,
                    run_inference,
                )

                def fake_popen(cmd, cwd=None):  # noqa: ARG001
                    # Strip the leading [python, -m, embedding_train.infer] prefix.
                    infer_argv = cmd[3:]
                    infer_args = infer_arg_parser().parse_args(infer_argv)
                    # Force CPU so the test works without CUDA.
                    infer_args.device = "cpu"
                    run_inference(infer_args)

                    class _Done:
                        def wait(self):
                            return 0

                    return _Done()

                popen_mock.side_effect = fake_popen
                run_parallel(args)

            embeddings_dir = output_root / "embeddings"
            shard_files = sorted(embeddings_dir.glob("shard_*.parquet"))
            self.assertEqual(len(shard_files), 2)

            total_rows = 0
            row_numbers = []
            for path in shard_files:
                table = pq.read_table(path)
                total_rows += table.num_rows
                row_numbers.extend(table.column("row_number").to_pylist())
            self.assertEqual(total_rows, 12)
            self.assertEqual(sorted(row_numbers), list(range(12)))

            first_table = pq.read_table(shard_files[0])
            embedding_field = first_table.schema.field("offer_embedding")
            # default precision is float16 in infer.py, but this test
            # overrode it to float32 above.
            self.assertEqual(
                str(embedding_field.type), "list<element: float>"
            )


if __name__ == "__main__":
    unittest.main()
