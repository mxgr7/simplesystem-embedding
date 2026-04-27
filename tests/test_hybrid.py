"""Unit tests for the hybrid search module — classifier patterns, RRF
fusion, and the run_search orchestrator with stubbed clients/embed."""

from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

# search-api is a flat directory, not a package — make it importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "search-api"))

from hybrid import (  # noqa: E402
    Hit,
    Mode,
    SearchParams,
    is_strict_identifier,
    rrf_merge,
    run_search,
)


# ────────────────────────────────────────────────────────────────────────
# Classifier
# ────────────────────────────────────────────────────────────────────────

class IsStrictIdentifierTests(unittest.TestCase):
    """Patterns lifted verbatim from hybrid_v0.md §"Query classifier"."""

    # (query, expected_classification)
    POSITIVES = [
        ("12345678", True),                # EAN-8
        ("4031100000000", True),           # EAN-13
        ("123456789012", True),            # UPC-A
        ("12345678901234", True),          # GTIN-14
        ("tze-231", True),                 # hyphenated, 7 chars, 3 digits
        ("221-413", True),                 # hyphenated, 7 chars, 6 digits
        ("gtb6-p5211", True),              # hyphenated, 10 chars, 5 digits
        ("e1987303", True),                # alpha-then-digit, 8 chars, 7 digits
        ("TZE-231", True),                 # case-insensitive
    ]

    NEGATIVES = [
        ("post-it", False),                # hyphenated, no digit
        ("t-shirt", False),                # hyphenated, no digit
        ("u-power", False),                # hyphenated, no digit
        ("o-ringe", False),                # hyphenated, no digit
        ("uni-ball", False),               # hyphenated, no digit
        ("magnet", False),                 # pure letters
        ("atlas", False),                  # pure letters
        ("kugelschreiber", False),         # pure letters
        ("114150", False),                 # 6 digits — ambiguous, not strict
        ("12345", False),                  # 5 digits — ambiguous, not strict
        ("123456789", False),              # 9 digits — ambiguous, not strict
        ("12345678901", False),            # 11 digits — ambiguous, not strict
        ("rj4", False),                    # below length floor
        ("ab1", False),                    # below length floor
        ("", False),                       # empty
        ("   ", False),                    # whitespace only
        ("hello world", False),            # contains space
        ("a" * 41, False),                 # over length cap
        # Industry-generic tokens — pre-tightening these were classified
        # strict, but the right answer is a dense+BM25 hybrid.
        ("rj45", False),                   # 4 chars — fails length ≥7
        ("RJ45", False),                   # case-insensitive variant
        ("lr44", False),                   # fails length ≥7
        ("ffp2", False),                   # fails length ≥7
        ("cr2032", False),                 # 6 chars, fails length ≥7
        ("dtw300", False),                 # 6 chars, fails length ≥7
        ("h07v-k", False),                 # 6 chars, fails length ≥7
        ("wd-40", False),                  # 5 chars, fails length ≥7
        # Denylist hits (would otherwise satisfy regex shape).
        ("usb-c", False),                  # short generic — denylist
        ("displayport", False),            # generic — no digits but defense-in-depth
    ]

    def test_positives(self):
        for q, expected in self.POSITIVES:
            with self.subTest(q=q):
                self.assertEqual(is_strict_identifier(q), expected)

    def test_negatives(self):
        for q, expected in self.NEGATIVES:
            with self.subTest(q=q):
                self.assertEqual(is_strict_identifier(q), expected)

    def test_strips_surrounding_whitespace(self):
        self.assertTrue(is_strict_identifier("  e1987303  "))


# ────────────────────────────────────────────────────────────────────────
# RRF
# ────────────────────────────────────────────────────────────────────────

class RrfMergeTests(unittest.TestCase):

    def test_single_list_preserves_order(self):
        merged = rrf_merge([[("a", 1.0), ("b", 0.5)]], k=60, top_n=10)
        self.assertEqual([hid for hid, _ in merged], ["a", "b"])

    def test_two_lists_top_one_in_both_wins(self):
        # Both lists rank "a" first → "a" must beat anything ranked second.
        dense = [("a", 0.9), ("b", 0.7), ("c", 0.5)]
        codes = [("a", 5.0), ("d", 4.0)]
        merged = rrf_merge([dense, codes], k=60, top_n=10)
        self.assertEqual(merged[0][0], "a")

    def test_codes_top_one_outranks_dense_rank_two(self):
        # Codes rank-1 (1/61) beats dense rank-2 (1/62); dense rank-1 ("d1")
        # is also at 1/61, so the top two share a fused score and the
        # tiebreak is id-ascending: "c1" < "d1".
        dense = [("d1", 0.9), ("d2", 0.85)]
        codes = [("c1", 10.0)]
        merged = rrf_merge([dense, codes], k=60, top_n=10)
        ids = [hid for hid, _ in merged]
        self.assertEqual(ids[:2], ["c1", "d1"])

    def test_ties_broken_by_id_ascending(self):
        # Two ids appearing only at rank 1 in their respective lists tie
        # at 1/(60+1). Without a deterministic tiebreak, dict insertion
        # order would decide. With id-asc, "alpha" < "beta".
        merged = rrf_merge([[("beta", 0.9)], [("alpha", 5.0)]], k=60, top_n=10)
        self.assertEqual([hid for hid, _ in merged], ["alpha", "beta"])

    def test_deterministic_across_leg_ordering(self):
        # Given the caller contract (each leg score-desc, id-asc on ties),
        # swapping the order of the leg lists at the merge call must not
        # change the output. The id-asc final tiebreak makes the merge
        # commutative over the leg axis when fused scores tie.
        dense = [("d1", 0.9), ("d2", 0.85)]
        codes = [("a", 5.0), ("z", 4.0)]
        m1 = rrf_merge([dense, codes], k=60, top_n=10)
        m2 = rrf_merge([codes, dense], k=60, top_n=10)
        self.assertEqual([h for h, _ in m1], [h for h, _ in m2])
        # Tied rank-1 hits across legs resolve "a" before "d1".
        self.assertEqual([h for h, _ in m1][:2], ["a", "d1"])

    def test_top_n_truncates(self):
        dense = [(f"d{i}", 1.0 - i * 0.01) for i in range(50)]
        merged = rrf_merge([dense], k=60, top_n=10)
        self.assertEqual(len(merged), 10)

    def test_empty_lists(self):
        self.assertEqual(rrf_merge([[], []], k=60, top_n=10), [])


# ────────────────────────────────────────────────────────────────────────
# run_search — stubbed clients
# ────────────────────────────────────────────────────────────────────────

def _make_dense_client(ids: list[tuple[str, float]]) -> MagicMock:
    """Stub MilvusClient.search returning the given dense hits."""
    c = MagicMock()
    c.search.return_value = [[
        {"distance": score, "entity": {"id": hid}}
        for hid, score in ids
    ]]
    return c


def _make_codes_client(ids: list[tuple[str, float]]) -> MagicMock:
    c = MagicMock()
    c.search.return_value = [[
        {"distance": score, "entity": {"id": hid}}
        for hid, score in ids
    ]]
    return c


async def _embed(_q: str) -> list[float]:
    return [0.0] * 128


class RunSearchTests(unittest.IsolatedAsyncioTestCase):

    async def test_vector_only_calls_dense_skips_codes(self):
        dense = _make_dense_client([("a", 0.9), ("b", 0.8)])
        codes = _make_codes_client([])
        hits, debug = await run_search(
            "test query",
            SearchParams(mode=Mode.VECTOR, k=10),
            dense_client=dense, codes_client=codes, embed=_embed,
        )
        self.assertEqual([h.id for h in hits], ["a", "b"])
        self.assertTrue(all(h.source == "dense" for h in hits))
        self.assertEqual(debug["path"], "vector")
        codes.search.assert_not_called()

    async def test_bm25_only_calls_codes_skips_dense(self):
        dense = _make_dense_client([])
        codes = _make_codes_client([("c1", 5.0), ("c2", 4.0)])
        hits, debug = await run_search(
            "rj45",
            SearchParams(mode=Mode.BM25, k=10),
            dense_client=dense, codes_client=codes, embed=_embed,
        )
        self.assertEqual([h.id for h in hits], ["c1", "c2"])
        self.assertTrue(all(h.source == "bm25" for h in hits))
        self.assertEqual(debug["path"], "bm25")
        dense.search.assert_not_called()

    async def test_hybrid_runs_both_legs(self):
        dense = _make_dense_client([("d1", 0.9), ("d2", 0.8)])
        codes = _make_codes_client([("c1", 5.0)])
        hits, debug = await run_search(
            "free text query",
            SearchParams(mode=Mode.HYBRID, k=10),
            dense_client=dense, codes_client=codes, embed=_embed,
        )
        ids = [h.id for h in hits]
        self.assertIn("d1", ids)
        self.assertIn("c1", ids)
        self.assertTrue(all(h.source == "rrf" for h in hits))
        self.assertEqual(debug["path"], "hybrid")
        # Hybrid never consults the classifier.
        self.assertIsNone(debug["classifier_strict"])

    async def test_hybrid_classified_strict_query_calls_only_codes(self):
        dense = _make_dense_client([])
        codes = _make_codes_client([("c1", 5.0), ("c2", 5.0)])
        hits, debug = await run_search(
            "e1987303",
            SearchParams(mode=Mode.HYBRID_CLASSIFIED, k=10),
            dense_client=dense, codes_client=codes, embed=_embed,
        )
        self.assertEqual([h.id for h in hits], ["c1", "c2"])
        self.assertTrue(all(h.source == "bm25" for h in hits))
        self.assertEqual(debug["path"], "strict")
        self.assertTrue(debug["classifier_strict"])
        self.assertFalse(debug["fallback_fired"])
        dense.search.assert_not_called()

    async def test_hybrid_classified_freetext_runs_both(self):
        dense = _make_dense_client([("d1", 0.9)])
        codes = _make_codes_client([])
        hits, debug = await run_search(
            "kugelschreiber",  # pure letters → not strict
            SearchParams(mode=Mode.HYBRID_CLASSIFIED, k=10),
            dense_client=dense, codes_client=codes, embed=_embed,
        )
        self.assertEqual(debug["path"], "hybrid")
        self.assertFalse(debug["classifier_strict"])
        self.assertEqual([h.id for h in hits], ["d1"])

    async def test_hybrid_classified_multiword_goes_vector_only(self):
        dense = _make_dense_client([("d1", 0.9), ("d2", 0.8)])
        codes = _make_codes_client([("c1", 5.0)])
        hits, debug = await run_search(
            "blue ballpoint pen",
            SearchParams(mode=Mode.HYBRID_CLASSIFIED, k=10),
            dense_client=dense, codes_client=codes, embed=_embed,
        )
        self.assertEqual([h.id for h in hits], ["d1", "d2"])
        self.assertTrue(all(h.source == "dense" for h in hits))
        self.assertEqual(debug["path"], "vector")
        # Classifier is short-circuited before it runs.
        self.assertIsNone(debug["classifier_strict"])
        codes.search.assert_not_called()

    async def test_strict_zero_results_falls_back_to_hybrid(self):
        dense = _make_dense_client([("d1", 0.9)])
        # First call (strict, large limit) returns []; second call (hybrid
        # codes_limit) returns []. We can't easily distinguish them with a
        # single mock, so set both to []; the dense leg supplies results.
        codes = _make_codes_client([])
        hits, debug = await run_search(
            "e1987303",
            SearchParams(mode=Mode.HYBRID_CLASSIFIED, k=10, enable_fallback=True),
            dense_client=dense, codes_client=codes, embed=_embed,
        )
        self.assertEqual(debug["path"], "fallback")
        self.assertTrue(debug["fallback_fired"])
        self.assertEqual([h.id for h in hits], ["d1"])

    async def test_strict_zero_results_without_fallback_returns_empty(self):
        dense = _make_dense_client([("d1", 0.9)])
        codes = _make_codes_client([])
        hits, debug = await run_search(
            "e1987303",
            SearchParams(mode=Mode.HYBRID_CLASSIFIED, k=10, enable_fallback=False),
            dense_client=dense, codes_client=codes, embed=_embed,
        )
        self.assertEqual(hits, [])
        self.assertFalse(debug["fallback_fired"])
        dense.search.assert_not_called()  # fallback was disabled

    async def test_empty_query_returns_empty(self):
        dense = _make_dense_client([("d1", 0.9)])
        codes = _make_codes_client([("c1", 5.0)])
        hits, _ = await run_search(
            "   ",
            SearchParams(mode=Mode.HYBRID_CLASSIFIED),
            dense_client=dense, codes_client=codes, embed=_embed,
        )
        self.assertEqual(hits, [])
        dense.search.assert_not_called()
        codes.search.assert_not_called()


if __name__ == "__main__":
    unittest.main()
