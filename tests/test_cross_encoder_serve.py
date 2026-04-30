"""Unit tests for cross_encoder_serve helpers.

The model-loading path is integration-tested manually (4GB checkpoint, GPU); these
tests cover the pure-Python feature pipeline that runs inside _lgbm_predict so a
schema or list-feature regression can't slip into prod silently.
"""
from __future__ import annotations

import math
import unittest

from omegaconf import OmegaConf

from cross_encoder_serve.inference import (
    _add_list_features,
    _char3_jaccard,
    _digit_jaccard,
    _engineered_features,
    _serving_features_cfg,
    _substring,
    _token_to_state,
)
from cross_encoder_train.features import FeatureExtractor


class TokenParsingTests(unittest.TestCase):
    def test_simple_state(self):
        self.assertEqual(_token_to_state("[EAN_MATCH]"), "MATCH")
        self.assertEqual(_token_to_state("[EAN_NONE]"), "NONE")
        self.assertEqual(_token_to_state("[EAN_MISMATCH]"), "MISMATCH")

    def test_compound_state(self):
        # State names with underscores must be preserved.
        self.assertEqual(_token_to_state("[ART_SUBSTRING_ONLY]"), "SUBSTRING_ONLY")
        self.assertEqual(_token_to_state("[ART_OFFER_INVALID]"), "OFFER_INVALID")


class LexicalFeatureTests(unittest.TestCase):
    def test_substring_case_insensitive(self):
        # "TE 6-A22" appears verbatim (case-insensitive) inside the offer name.
        self.assertEqual(_substring("TE 6-A22", "Hilti Akku-Bohrhammer TE 6-A22"), 1)
        # Mixed case query, mixed case offer — still matches because both sides lowercased.
        self.assertEqual(_substring("hilti", "Hilti Akku-Bohrhammer"), 1)
        self.assertEqual(_substring("Bosch 18V", "Hilti Akku-Bohrhammer"), 0)

    def test_substring_empty(self):
        self.assertEqual(_substring("", "anything"), 0)
        self.assertEqual(_substring("anything", ""), 0)

    def test_digit_jaccard(self):
        self.assertEqual(_digit_jaccard("Hilti TE 6 A22", "Hilti TE 6 A22"), 1.0)
        # "6" and "22" overlap; query has {6,22}, offer has {6,18,22} → |∩|=2 |∪|=3
        self.assertAlmostEqual(_digit_jaccard("TE 6 A22", "TE 6 18 A22"), 2.0 / 3.0)

    def test_digit_jaccard_no_digits(self):
        self.assertEqual(_digit_jaccard("hilti", "bosch"), 0.0)

    def test_char3_jaccard_identical(self):
        # 3-grams of "abcd" = {abc, bcd} (lowercased)
        self.assertEqual(_char3_jaccard("abcd", "abcd"), 1.0)

    def test_char3_jaccard_disjoint(self):
        self.assertEqual(_char3_jaccard("abc", "xyz"), 0.0)


class EngineeredFeatureTests(unittest.TestCase):
    def setUp(self):
        self.extractor = FeatureExtractor(_serving_features_cfg())

    def test_ean_match_one_hot(self):
        # Query contains a valid 13-digit GTIN; offer ean matches.
        ctx = {
            "query_term": "buy ean 4006381333931 please",
            "ean": "4006381333931",
            "article_number": None,
            "manufacturer_article_number": None,
        }
        feats = _engineered_features(self.extractor, ctx, ctx["query_term"], "Some name")
        self.assertEqual(feats["ean_MATCH"], 1)
        self.assertEqual(feats["ean_NONE"], 0)
        self.assertEqual(feats["ean_MISMATCH"], 0)

    def test_ean_none_when_query_has_no_gtin(self):
        ctx = {"query_term": "Hilti drill", "ean": "4006381333931",
               "article_number": None, "manufacturer_article_number": None}
        feats = _engineered_features(self.extractor, ctx, ctx["query_term"], "")
        self.assertEqual(feats["ean_NONE"], 1)
        self.assertEqual(feats["ean_MATCH"], 0)

    def test_article_exact(self):
        ctx = {"query_term": "TE6A22 hilti", "ean": None,
               "article_number": "TE6A22", "manufacturer_article_number": None}
        feats = _engineered_features(self.extractor, ctx, ctx["query_term"], "")
        self.assertEqual(feats["art_EXACT"], 1)
        self.assertEqual(feats["art_NONE"], 0)

    def test_one_hot_rows_sum_to_one(self):
        ctx = {"query_term": "TE6A22", "ean": None,
               "article_number": "WRONG", "manufacturer_article_number": None}
        feats = _engineered_features(self.extractor, ctx, ctx["query_term"], "")
        ean_sum = feats["ean_NONE"] + feats["ean_MATCH"] + feats["ean_MISMATCH"]
        art_sum = (feats["art_NONE"] + feats["art_EXACT"]
                   + feats["art_SUBSTRING_ONLY"] + feats["art_MISMATCH"]
                   + feats["art_OFFER_INVALID"])
        self.assertEqual(ean_sum, 1)
        self.assertEqual(art_sum, 1)


class ListFeatureTests(unittest.TestCase):
    def _row(self, exact, sub, irr, comp=0.05):
        return {
            "ce_p_irrelevant": irr,
            "ce_p_complement": comp,
            "ce_p_substitute": sub,
            "ce_p_exact": exact,
        }

    def test_rank_and_gap_for_single_row(self):
        rows = _add_list_features([self._row(0.7, 0.2, 0.1)])
        self.assertEqual(rows[0]["ce_p_exact_rank_desc"], 1)
        self.assertAlmostEqual(rows[0]["ce_p_exact_gap_from_max"], 0.0)
        self.assertEqual(rows[0]["group_size"], 1)

    def test_rank_descending(self):
        rows = _add_list_features([
            self._row(0.9, 0.05, 0.05),
            self._row(0.5, 0.3, 0.2),
            self._row(0.1, 0.1, 0.8),
        ])
        # rank by p_exact descending: 0.9 → 1, 0.5 → 2, 0.1 → 3
        self.assertEqual(rows[0]["ce_p_exact_rank_desc"], 1)
        self.assertEqual(rows[1]["ce_p_exact_rank_desc"], 2)
        self.assertEqual(rows[2]["ce_p_exact_rank_desc"], 3)
        self.assertAlmostEqual(rows[0]["ce_p_exact_gap_from_max"], 0.0)
        self.assertAlmostEqual(rows[1]["ce_p_exact_gap_from_max"], 0.4)
        self.assertAlmostEqual(rows[2]["ce_p_exact_gap_from_max"], 0.8)

    def test_zscore_centered_when_uniform(self):
        # Identical p_exact across rows → z-score should be 0 (gstd guarded to 1.0).
        rows = _add_list_features([self._row(0.5, 0.3, 0.2) for _ in range(3)])
        for r in rows:
            self.assertAlmostEqual(r["ce_p_exact_zscore"], 0.0)

    def test_group_size_broadcast(self):
        rows = _add_list_features([
            self._row(0.9, 0.05, 0.05),
            self._row(0.5, 0.3, 0.2),
            self._row(0.1, 0.1, 0.8),
        ])
        for r in rows:
            self.assertEqual(r["group_size"], 3)

    def test_dense_rank_with_ties(self):
        # Two rows tie on p_exact; both should get rank 1, next gets rank 2.
        rows = _add_list_features([
            self._row(0.9, 0.05, 0.05),
            self._row(0.9, 0.1, 0.0),
            self._row(0.3, 0.5, 0.2),
        ])
        ranks = [r["ce_p_exact_rank_desc"] for r in rows]
        self.assertEqual(sorted(ranks), [1, 1, 2])


if __name__ == "__main__":
    unittest.main()
