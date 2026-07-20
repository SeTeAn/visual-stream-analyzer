from __future__ import annotations

import unittest

from stream_analysis.evaluation.review_contract import canonical_reviewer_id


class ReviewContractTests(unittest.TestCase):
    def test_accepts_the_shared_canonical_identifier_language(self) -> None:
        for value in ("reviewer-a", "reviewer_b", "r1", "a" * 64):
            with self.subTest(value=value):
                self.assertEqual(canonical_reviewer_id(value), value)

    def test_rejects_identifiers_that_could_diverge_between_stages(self) -> None:
        for value in (
            "reviewer.a",
            "Reviewer-a",
            " reviewer-a",
            "reviewer-a ",
            "",
            "a" * 65,
            None,
        ):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "reviewer"):
                    canonical_reviewer_id(value)


if __name__ == "__main__":
    unittest.main()
