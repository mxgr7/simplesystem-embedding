import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import faiss
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from embedding_train.index_build import build_arg_parser, run_index_build
from embedding_train.index_build_train import (
    build_arg_parser as train_arg_parser,
    train_index,
)
from embedding_train.index_merge import (
    build_arg_parser as merge_arg_parser,
    merge_shards,
)
from tests.test_index_build import _IndexModelStub, _TokenizerStub, build_cfg


class ShardMergeFlowTests(unittest.TestCase):
    def write_offer_table(self, path, names):
        table = pa.Table.from_pylist(
            [
                {"offer_id_b64": f"o{i}", "name": name, "label": "Exact"}
                for i, name in enumerate(names)
            ]
        )
        pq.write_table(table, path)

    @patch("embedding_train.infer.AutoTokenizer.from_pretrained")
    @patch("embedding_train.index_build_train.load_index_model")
    @patch("embedding_train.index_build.load_embedding_module_from_checkpoint")
    def test_train_then_shard_then_merge(
        self,
        load_checkpoint,
        load_index_model_mock,
        from_pretrained,
    ):
        cfg = build_cfg()
        load_checkpoint.return_value = (_IndexModelStub(), cfg)
        load_index_model_mock.return_value = (_IndexModelStub(), cfg)
        from_pretrained.return_value = _TokenizerStub()

        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            train_input = tmp_path / "train_sample.parquet"
            self.write_offer_table(
                train_input,
                ["bolt"] * 6 + ["screw"] * 6 + ["nut"] * 6,
            )
            shard0_input = tmp_path / "shard0.parquet"
            shard1_input = tmp_path / "shard1.parquet"
            self.write_offer_table(shard0_input, ["bolt", "screw", "nut", "bolt"])
            self.write_offer_table(shard1_input, ["nut", "screw", "bolt", "screw"])

            trained_index_path = tmp_path / "trained.index"
            train_args = train_arg_parser().parse_args(
                [
                    "--checkpoint",
                    str(tmp_path / "ckpt"),
                    "--input",
                    str(train_input),
                    "--output",
                    str(trained_index_path),
                    "--index-type",
                    "ivf_flat",
                    "--nlist",
                    "2",
                    "--train-sample-size",
                    "12",
                    "--read-batch-size",
                    "8",
                    "--encode-batch-size",
                    "8",
                ]
            )
            train_index(train_args)
            self.assertTrue(trained_index_path.exists())

            shard0_out = tmp_path / "shard0_artifact"
            shard1_out = tmp_path / "shard1_artifact"
            for shard_in, shard_out, offset in (
                (shard0_input, shard0_out, 0),
                (shard1_input, shard1_out, 4),
            ):
                build_args = build_arg_parser().parse_args(
                    [
                        "--checkpoint",
                        str(tmp_path / "ckpt"),
                        "--input",
                        str(shard_in),
                        "--output",
                        str(shard_out),
                        "--index-type",
                        "ivf_flat",
                        "--nlist",
                        "2",
                        "--trained-index",
                        str(trained_index_path),
                        "--faiss-id-offset",
                        str(offset),
                        "--read-batch-size",
                        "4",
                        "--encode-batch-size",
                        "4",
                    ]
                )
                run_index_build(build_args)

            merged_out = tmp_path / "merged_artifact"
            merge_args = merge_arg_parser().parse_args(
                [
                    "--shards",
                    str(shard0_out),
                    str(shard1_out),
                    "--output",
                    str(merged_out),
                ]
            )
            merge_shards(merge_args)

            merged_index = faiss.read_index(str(merged_out / "index.faiss"))
            self.assertEqual(merged_index.ntotal, 8)
            self.assertTrue(merged_index.is_trained)

            merged_metadata = pq.read_table(str(merged_out / "metadata.parquet"))
            self.assertEqual(merged_metadata.num_rows, 8)
            ids = np.asarray(merged_metadata.column("faiss_id").to_pylist())
            self.assertEqual(set(int(v) for v in ids), set(range(8)))

            merged_index.nprobe = 2
            query = np.array([[1.0, 0.0, 0.0]], dtype="float32")
            _, neighbors = merged_index.search(query, 3)
            self.assertEqual(neighbors.shape, (1, 3))
            # "bolt" vectors live at shard0 ids {0, 3} and shard1 id {6}
            top_ids = set(int(v) for v in neighbors[0])
            self.assertTrue(top_ids.intersection({0, 3, 6}))


if __name__ == "__main__":
    unittest.main()
