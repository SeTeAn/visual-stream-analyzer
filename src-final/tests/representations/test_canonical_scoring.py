import unittest

import numpy as np

from stream_analysis.representations import (
    DinoV2CosineScorer,
    HandcraftedCanonicalScorer,
    HandcraftedRepresentationConfig,
)

try:
    from matching._fixtures import candidate, representation
except ImportError:
    from _fixtures import candidate, representation

try:
    from .test_handcrafted import build_record
except ImportError:
    from test_handcrafted import build_record


class CanonicalScoringTest(unittest.TestCase):
    def test_dino_cosine_is_preserved_and_mapped_to_unit_interval(self):
        item = candidate("a", 0, 10)
        left = representation(item, (1.0, 0.0))
        same = representation(candidate("b", 1, 10), (1.0, 0.0))
        opposite = representation(candidate("c", 1, 10), (-1.0, 0.0))
        scorer = DinoV2CosineScorer()
        self.assertEqual(scorer.score(left, same).visual_score, 1.0)
        self.assertEqual(scorer.score(left, opposite).visual_score, 0.0)
        self.assertEqual(scorer.score(left, opposite).raw_metric_value, -1.0)

    def test_incompatible_variants_are_rejected(self):
        from dataclasses import replace

        left = representation(candidate("a", 0, 10))
        right = replace(representation(candidate("b", 1, 10)), input_variant="other")
        with self.assertRaisesRegex(ValueError, "variants"):
            DinoV2CosineScorer().score(left, right)

    def test_handcrafted_adapter_preserves_raw_distance_and_similarity(self):
        config = HandcraftedRepresentationConfig()
        mask = np.zeros((20, 20), dtype=bool)
        mask[3:17, 4:16] = True
        left = build_record(mask, config=config, foreground_color=(220, 40, 30))[0]
        right = build_record(mask, config=config, foreground_color=(30, 170, 210))[0]
        score = HandcraftedCanonicalScorer(config).score(left, right)
        self.assertEqual(score.raw_metric_name, "handcrafted_distance")
        self.assertAlmostEqual(score.visual_score, 1.0 - score.raw_metric_value, places=15)


if __name__ == "__main__":
    unittest.main()
