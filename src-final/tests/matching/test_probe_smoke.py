import unittest
from pathlib import Path

from stream_analysis import (
    CandidateExtractionConfig,
    DinoV2RepresentationConfig,
    FramePair,
    HandcraftedRepresentationConfig,
    ManifestLoadRequest,
    build_dinov2_representations,
    build_handcrafted_representations,
    extract_candidates,
    load_decoded_stream,
)
from stream_analysis.matching import (
    EventConfig,
    GroupingConfig,
    MatchingConfig,
    PairScoringConfig,
    build_change_events,
    group_recurring_visual_types,
    match_neighboring_frames,
)
from stream_analysis.representations import (
    DINO_BBOX_VARIANT,
    DINO_MASK_NEUTRAL_VARIANT,
    HANDCRAFTED_BBOX_VARIANT,
    HANDCRAFTED_MASK_VARIANT,
    DinoV2CosineScorer,
    HandcraftedCanonicalScorer,
)

from representations.test_dinov2_integration import FakeDinoProvider, config_for, input_producer


ROOTS = tuple(sorted(Path("src-final/data/streams").glob("probe_*")))


def _inventory(root):
    return tuple(sorted(path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()))


def _semantic_smoke(decoded, snapshot, records, scorer, variant):
    scoring = PairScoringConfig(
        scorer_id=scorer.scorer_id,
        scorer_version=scorer.scorer_version,
        visual_gate=0.0,
        spatial_gate=None,
    )
    matching_config = MatchingConfig(
        unmatched_pair_cost=0.5,
        local_margin_gate=0.0,
        global_margin_gate=0.0,
        severe_quality_flags=("POSSIBLE_MERGING", "POSSIBLE_SPLITTING"),
    )
    candidates = snapshot.result.candidates
    by_frame = {
        frame.frame_id: tuple(item for item in candidates if item.frame_id == frame.frame_id)
        for frame in decoded.stream.frames
    }
    matching = []
    for left, right in zip(decoded.stream.frames, decoded.stream.frames[1:]):
        matching.append(
            match_neighboring_frames(
                stream_id=decoded.stream.stream_id,
                frame_pair=FramePair(
                    from_frame_id=left.frame_id,
                    from_frame_index=left.index,
                    from_frame_size=left.image_size,
                    to_frame_id=right.frame_id,
                    to_frame_index=right.index,
                    to_frame_size=right.image_size,
                ),
                from_candidates=by_frame[left.frame_id],
                to_candidates=by_frame[right.frame_id],
                representations=records,
                representation_variant_id=variant,
                scorer=scorer,
                scoring_config=scoring,
                matching_config=matching_config,
            ).result
        )
    invalid_matching = sum(
        item.envelope.validity_status.value == "invalid" for item in matching
    )
    if invalid_matching:
        return {
            "candidates": len(candidates),
            "comparisons": len(matching),
            "types": None,
            "events": None,
            "withheld": invalid_matching,
            "grouping_completed": False,
            "failure_reason": "invalid_neighboring_representation_score",
        }
    try:
        grouping = group_recurring_visual_types(
            stream_id=decoded.stream.stream_id,
            candidates=candidates,
            representations=records,
            matching_results=matching,
            representation_variant_id=variant,
            scorer=scorer,
            config=GroupingConfig(
                medoid_gate=1.0,
                support_quantile=0.25,
                quantile_gate=1.0,
                support_pair_gate=1.0,
                support_ratio_gate=1.0,
                visual_gate=1.0,
                medoid_weight=0.4,
                quantile_weight=0.4,
                support_ratio_weight=0.2,
                second_best_margin=1.0,
                representation_variant_id=variant,
                scorer_id=scorer.scorer_id,
                scorer_version=scorer.scorer_version,
            ),
        )
    except ValueError as exc:
        return {
            "candidates": len(candidates),
            "comparisons": len(matching),
            "types": None,
            "events": None,
            "withheld": len(matching),
            "grouping_completed": False,
            "failure_reason": str(exc),
        }
    event_batch = build_change_events(
        stream_id=decoded.stream.stream_id,
        candidates=candidates,
        grouping=grouping,
        matching_results=matching,
        config=EventConfig(
            position_threshold_norm=0.1,
            severe_quality_flags=("POSSIBLE_MERGING", "POSSIBLE_SPLITTING"),
        ),
    )
    return {
        "candidates": len(candidates),
        "comparisons": len(matching),
        "types": len(grouping.recurring_types),
        "events": len(event_batch.events),
        "withheld": sum(item.emission_status.value == "withheld" for item in event_batch.comparisons),
        "grouping_completed": True,
        "failure_reason": None,
    }


def run_probe_smoke():
    summaries = []
    producer = input_producer()
    extraction_config = CandidateExtractionConfig()
    for root in ROOTS:
        decoded = load_decoded_stream(ManifestLoadRequest(stream_root=root, producer=producer))
        snapshot = extract_candidates(decoded, extraction_config)
        for variant in (HANDCRAFTED_MASK_VARIANT, HANDCRAFTED_BBOX_VARIANT):
            config = HandcraftedRepresentationConfig(variant=variant)
            batch = build_handcrafted_representations(decoded, snapshot, config)
            summary = _semantic_smoke(
                decoded, snapshot, batch.records, HandcraftedCanonicalScorer(config), variant
            )
            summaries.append((root.name, variant, summary))
        for variant in (DINO_BBOX_VARIANT, DINO_MASK_NEUTRAL_VARIANT):
            provider = FakeDinoProvider(batch_size=17)
            config = config_for(provider, variant=variant)
            batch = build_dinov2_representations(decoded, snapshot, provider, config)
            summary = _semantic_smoke(
                decoded, snapshot, batch.records, DinoV2CosineScorer(), variant
            )
            summaries.append((root.name, variant, summary))
    return tuple(summaries)


class ProbePipelineSmokeTest(unittest.TestCase):
    def test_read_only_all_probes_and_representation_variants(self):
        before_data = _inventory(Path("src-final/data"))
        before_outputs = _inventory(Path("src-final/outputs"))
        summaries = run_probe_smoke()
        self.assertEqual(len(summaries), 16)
        for stream, variant, summary in summaries:
            with self.subTest(stream=stream, variant=variant):
                self.assertGreater(summary["candidates"], 0)
                self.assertGreater(summary["comparisons"], 0)
                self.assertTrue(summary["grouping_completed"])
                self.assertEqual(summary["withheld"], 0)
        self.assertEqual(before_data, _inventory(Path("src-final/data")))
        self.assertEqual(before_outputs, _inventory(Path("src-final/outputs")))


if __name__ == "__main__":
    unittest.main()
