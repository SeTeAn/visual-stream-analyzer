from __future__ import annotations

import unittest

from stream_analysis.evaluation import PairRankingRecord, evaluate_pair_ranking


class StrictRankingTest(unittest.TestCase):
    def test_invalid_negative_ranks_above_valid_and_invalid_positive_below(self) -> None:
        result = evaluate_pair_ranking((
            PairRankingRecord("q", "bad_invalid", False, None, valid=False),
            PairRankingRecord("q", "good_valid", True, 1.0, valid=True),
            PairRankingRecord("q", "bad_valid", False, 0.0, valid=True),
            PairRankingRecord("q", "good_invalid", True, None, valid=False),
        ))
        self.assertEqual(result["status"], "supported")
        self.assertLess(result["average_precision"], 1.0)
        self.assertEqual(result["retrieval"]["forward"]["recall@1"], 0.0)
        self.assertEqual(result["retrieval"]["reverse"]["recall@1"], 1.0)
        self.assertEqual(result["retrieval"]["aggregate"]["recall@1"], 0.5)
        self.assertEqual(result["invalid_pair_count"], 2)

    def test_valid_score_ties_do_not_depend_on_candidate_ids_or_input_order(self) -> None:
        first = evaluate_pair_ranking((
            PairRankingRecord("q", "a_positive", True, 0.5, valid=True),
            PairRankingRecord("q", "z_negative", False, 0.5, valid=True),
        ))
        renamed = evaluate_pair_ranking((
            PairRankingRecord("q", "a_negative", False, 0.5, valid=True),
            PairRankingRecord("q", "z_positive", True, 0.5, valid=True),
        ))
        reversed_order = evaluate_pair_ranking((
            PairRankingRecord("q", "z_negative", False, 0.5, valid=True),
            PairRankingRecord("q", "a_positive", True, 0.5, valid=True),
        ))

        for result in (first, renamed, reversed_order):
            self.assertEqual(result["status"], "supported")
            self.assertAlmostEqual(result["average_precision"], 0.75)
            self.assertAlmostEqual(result["auroc"], 0.5)
            self.assertAlmostEqual(result["retrieval"]["forward"]["map"], 0.75)
            self.assertAlmostEqual(result["retrieval"]["forward"]["mrr"], 2 / 3)
            self.assertEqual(result["retrieval"]["forward"]["recall@1"], 0.0)
            self.assertEqual(result["retrieval"]["forward"]["recall@3"], 1.0)
            self.assertAlmostEqual(result["retrieval"]["reverse"]["map"], 1.0)
            self.assertAlmostEqual(result["retrieval"]["aggregate"]["map"], 0.875)
            self.assertAlmostEqual(result["retrieval"]["map"], 0.875)

    def test_invalid_query_gets_zero_metrics_in_its_direction(self) -> None:
        result = evaluate_pair_ranking((
            PairRankingRecord(
                "invalid_query", "positive", True, None,
                valid=False, query_valid=False, gallery_valid=True,
            ),
            PairRankingRecord(
                "invalid_query", "negative", False, None,
                valid=False, query_valid=False, gallery_valid=True,
            ),
        ))
        forward = result["retrieval"]["forward"]
        self.assertEqual(forward["query_count"], 1)
        self.assertEqual(forward["invalid_query_count"], 1)
        self.assertEqual(forward["map"], 0.0)
        self.assertEqual(forward["mrr"], 0.0)
        self.assertEqual(forward["recall@1"], 0.0)
        self.assertEqual(forward["per_query"][0]["query_valid"], False)
        self.assertEqual(result["invalid_pair_count"], 2)

    def test_valid_only_retrieval_and_coverages_are_published(self) -> None:
        result = evaluate_pair_ranking(
            (
                PairRankingRecord("q_valid", "g_positive", True, 0.9),
                PairRankingRecord("q_valid", "g_negative", False, 0.1),
                PairRankingRecord(
                    "q_invalid", "g_positive", True, None,
                    valid=False, query_valid=False,
                ),
                PairRankingRecord(
                    "q_invalid", "g_negative", False, None,
                    valid=False, query_valid=False,
                ),
            ),
            candidate_validity={
                "q_valid": True,
                "q_invalid": False,
                "g_positive": True,
                "g_negative": True,
            },
        )
        valid_only = result["valid_only"]
        self.assertEqual(valid_only["candidate_coverage"], 0.75)
        self.assertEqual(valid_only["pair_coverage"], 0.5)
        self.assertEqual(valid_only["average_precision"], 1.0)
        self.assertEqual(valid_only["auroc"], 1.0)
        self.assertEqual(
            valid_only["retrieval"]["forward"]["query_coverage"],
            0.5,
        )
        self.assertEqual(
            valid_only["retrieval"]["reverse"]["query_coverage"],
            1.0,
        )
        self.assertAlmostEqual(
            valid_only["retrieval"]["aggregate"]["query_coverage"],
            2 / 3,
        )
        self.assertEqual(valid_only["retrieval"]["forward"]["map"], 1.0)

    def test_pair_validity_requires_matching_visual_score_presence(self) -> None:
        with self.assertRaises(ValueError):
            PairRankingRecord("q", "g", True, None, valid=True)
        with self.assertRaises(ValueError):
            PairRankingRecord("q", "g", True, 0.5, valid=False)


if __name__ == "__main__":
    unittest.main()
