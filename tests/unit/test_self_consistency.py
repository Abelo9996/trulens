"""Unit tests for trulens.feedback.self_consistency.SelfConsistency."""

from __future__ import annotations

import inspect
import unittest
from unittest.mock import MagicMock

from trulens.feedback.self_consistency import SelfConsistency


def _mock_relevance(prompt: str, response: str) -> float: ...


def _sequence_provider(scores, reason=None):
    """A provider whose relevance() returns the next score on each call."""
    provider = MagicMock()
    provider.model_engine = "stub"
    it = iter(scores)
    if reason is None:
        provider.relevance.side_effect = lambda prompt, response, **kw: next(it)
    else:
        provider.relevance.side_effect = lambda prompt, response, **kw: (
            next(it),
            {"reason": reason},
        )
    provider.relevance.__signature__ = inspect.signature(_mock_relevance)
    return provider


class TestSelfConsistencyConstruction(unittest.TestCase):
    def test_rejects_non_positive_trials(self):
        provider = _sequence_provider([1.0])
        with self.assertRaises(ValueError):
            SelfConsistency(provider, "relevance", n_trials=0)

    def test_rejects_unknown_aggregation(self):
        provider = _sequence_provider([1.0])
        with self.assertRaises(ValueError):
            SelfConsistency(provider, "relevance", aggregation="nonsense")

    def test_rejects_missing_method(self):
        provider = MagicMock(spec=[])
        with self.assertRaises(AttributeError):
            SelfConsistency(provider, "relevance")

    def test_exposes_wrapped_signature(self):
        provider = _sequence_provider([1.0])
        judge = SelfConsistency(provider, "relevance", n_trials=3)
        self.assertEqual(
            list(inspect.signature(judge).parameters),
            ["prompt", "response"],
        )


class TestSelfConsistencyAggregation(unittest.TestCase):
    def test_mean(self):
        judge = SelfConsistency(
            _sequence_provider([0.0, 0.5, 1.0]),
            "relevance",
            n_trials=3,
            aggregation="mean",
        )
        score, _ = judge("q", "a")
        self.assertAlmostEqual(score, 0.5)

    def test_median(self):
        judge = SelfConsistency(
            _sequence_provider([0.1, 0.9, 1.0]),
            "relevance",
            n_trials=3,
            aggregation="median",
        )
        score, _ = judge("q", "a")
        self.assertAlmostEqual(score, 0.9)

    def test_majority_vote_passes(self):
        judge = SelfConsistency(
            _sequence_provider([1.0, 1.0, 0.0]),
            "relevance",
            n_trials=3,
            aggregation="majority_vote",
        )
        score, _ = judge("q", "a")
        self.assertEqual(score, 1.0)

    def test_callable_aggregation(self):
        judge = SelfConsistency(
            _sequence_provider([0.2, 0.4, 0.6]),
            "relevance",
            n_trials=3,
            aggregation=max,
        )
        score, _ = judge("q", "a")
        self.assertAlmostEqual(score, 0.6)


class TestSelfConsistencyReliability(unittest.TestCase):
    def test_perfect_consistency_has_zero_flip_rate(self):
        judge = SelfConsistency(
            _sequence_provider([1.0, 1.0, 1.0, 1.0]),
            "relevance",
            n_trials=4,
        )
        _, meta = judge("q", "a")
        rel = meta["self_consistency"]
        self.assertEqual(rel["flip_rate"], 0.0)
        self.assertEqual(rel["outcome_entropy"], 0.0)
        self.assertEqual(rel["score_std"], 0.0)
        self.assertEqual(rel["n_trials"], 4)

    def test_even_split_is_maximally_unstable(self):
        # Two pass, two fail -> flip rate 0.5, entropy 1 bit.
        judge = SelfConsistency(
            _sequence_provider([1.0, 1.0, 0.0, 0.0]),
            "relevance",
            n_trials=4,
        )
        _, meta = judge("q", "a")
        rel = meta["self_consistency"]
        self.assertAlmostEqual(rel["flip_rate"], 0.5)
        self.assertAlmostEqual(rel["agreement"], 0.5)
        self.assertAlmostEqual(rel["outcome_entropy"], 1.0)
        self.assertGreater(rel["score_std"], 0.0)

    def test_reason_and_scores_recorded(self):
        judge = SelfConsistency(
            _sequence_provider([0.2, 0.8], reason="because"),
            "relevance",
            n_trials=2,
        )
        _, meta = judge("q", "a")
        self.assertIn("reason", meta)
        self.assertIn("because", meta["reason"])
        self.assertEqual(sorted(meta["self_consistency"]["scores"]), [0.2, 0.8])


if __name__ == "__main__":
    unittest.main()
