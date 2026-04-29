"""SentencePiece BPE tokenizer for the suggest LM.

The corpus is the training-target frequency table, with each target
replicated by its event count (capped to ``max_repeat`` to keep ultra-head
items from dominating). This is a reasonable proxy for sampling the LM's
training distribution when learning subword merges.

Specials:
    <pad> = 0   (left-pad / batch padding)
    <unk> = 1
    <s>   = 2   (begin-of-sequence)
    </s>  = 3   (end-of-sequence)
    <sep>     -- prefix/completion separator (used by the prefix-conditioned
                 training option in the plan; included now so we don't have
                 to retrain the tokenizer to enable it later).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import duckdb

from .data import TARGETS_DIR, TOKENIZER_DIR

DEFAULT_VOCAB_SIZE = 8000
DEFAULT_MAX_REPEAT = 100


class Tokenizer:
    """Thin wrapper around a SentencePiece processor with id constants."""

    def __init__(self, sp) -> None:
        self.sp = sp
        self.pad_id = sp.pad_id()
        self.unk_id = sp.unk_id()
        self.bos_id = sp.bos_id()
        self.eos_id = sp.eos_id()
        self.sep_id = sp.piece_to_id("<sep>")
        if self.sep_id == self.unk_id:
            raise ValueError("Tokenizer is missing the <sep> special token.")
        self.vocab_size = sp.vocab_size()

    @classmethod
    def load(cls, path: Path) -> "Tokenizer":
        import sentencepiece as spm

        sp = spm.SentencePieceProcessor()
        sp.Load(str(path))
        return cls(sp)

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[int]:
        ids = self.sp.encode(text, out_type=int)
        if add_bos:
            ids = [self.bos_id, *ids]
        if add_eos:
            ids = [*ids, self.eos_id]
        return ids

    def decode(self, ids) -> str:
        return self.sp.decode(list(ids))

    def piece_to_id(self, piece: str) -> int:
        return self.sp.piece_to_id(piece)

    def id_to_piece(self, idx: int) -> str:
        return self.sp.id_to_piece(idx)


def train(
    out_dir: Path = TOKENIZER_DIR,
    targets_dir: Path = TARGETS_DIR,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    max_repeat: int = DEFAULT_MAX_REPEAT,
) -> Path:
    import sentencepiece as spm

    out_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = out_dir / "corpus.txt"

    glob = f"{targets_dir}/split=train/**/*.parquet"
    print(f"[1/3] Aggregating train targets from {glob}", flush=True)
    con = duckdb.connect(":memory:")
    df = con.execute(
        f"""
        SELECT target, sum(count)::BIGINT AS count
        FROM read_parquet('{glob}', hive_partitioning = TRUE)
        GROUP BY target
        """
    ).fetchdf()

    print(
        f"      unique targets: {len(df):,}  "
        f"sum(count): {int(df['count'].sum()):,}",
        flush=True,
    )

    print(f"[2/3] Writing weighted corpus to {corpus_path} "
          f"(max_repeat={max_repeat})...", flush=True)
    t0 = time.time()
    n_lines = 0
    with corpus_path.open("w", encoding="utf-8") as f:
        for target, count in zip(df["target"], df["count"]):
            line = target.replace("\n", " ").replace("\r", " ").strip()
            if not line:
                continue
            reps = min(int(count), max_repeat)
            for _ in range(reps):
                f.write(line)
                f.write("\n")
                n_lines += 1
    print(
        f"      wrote {n_lines:,} lines, "
        f"{corpus_path.stat().st_size / (1024*1024):.1f} MB  "
        f"({time.time()-t0:.1f}s)",
        flush=True,
    )

    model_prefix = out_dir / "spm"
    print(f"[3/3] Training SentencePiece BPE (vocab={vocab_size})...",
          flush=True)
    t0 = time.time()
    spm.SentencePieceTrainer.train(
        input=str(corpus_path),
        model_prefix=str(model_prefix),
        vocab_size=vocab_size,
        model_type="bpe",
        character_coverage=0.9999,
        normalization_rule_name="identity",
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        pad_piece="<pad>",
        unk_piece="<unk>",
        bos_piece="<s>",
        eos_piece="</s>",
        user_defined_symbols=["<sep>"],
        # Process all lines; corpus is small enough that we don't need
        # SentencePiece's own subsampling.
        input_sentence_size=0,
        shuffle_input_sentence=True,
        num_threads=8,
    )
    print(f"      tokenizer trained in {time.time()-t0:.1f}s", flush=True)
    print(f"      model: {model_prefix}.model", flush=True)
    print(f"      vocab: {model_prefix}.vocab", flush=True)
    return Path(f"{model_prefix}.model")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=TOKENIZER_DIR)
    p.add_argument("--targets-dir", type=Path, default=TARGETS_DIR)
    p.add_argument("--vocab-size", type=int, default=DEFAULT_VOCAB_SIZE)
    p.add_argument("--max-repeat", type=int, default=DEFAULT_MAX_REPEAT,
                   help="Cap per-target replication when building the "
                        "weighted corpus.")
    args = p.parse_args()

    model_path = train(
        out_dir=args.out_dir,
        targets_dir=args.targets_dir,
        vocab_size=args.vocab_size,
        max_repeat=args.max_repeat,
    )

    tok = Tokenizer.load(model_path)
    samples = ["schraube", "schrauben", "kugelschreiber", "Pflaster",
               "31060", "Spannfutter dreibacken 125mm"]
    print("\nSanity check:")
    for s in samples:
        ids = tok.encode(s)
        pieces = [tok.id_to_piece(i) for i in ids]
        print(f"  {s!r:40} -> {pieces}")


if __name__ == "__main__":
    main()
