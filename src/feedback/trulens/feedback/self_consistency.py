"""Self-consistency — repeated-trial reliability for a single LLM judge.

A single LLM judge is a one-shot draw. At non-zero temperature the same judge,
given identical inputs, does not always return the same verdict, and the
evaluation pipeline records only that one draw. ``SelfConsistency`` runs the
same judge ``n_trials`` times, aggregates the repeated trials into one score,
and attaches a reliability summary (dispersion, and for a pass/fail view the
flip rate and outcome entropy) so a caller can see how stable that score was.

This is complementary to :class:`~trulens.feedback.jury.Jury`. ``Jury``
ensembles *different* judges to reduce inter-judge bias; ``SelfConsistency``
repeats *one* judge to measure and reduce intra-judge, run-to-run variance.

The reliability metrics follow the definitions in "The Coin Flip Judge?
Reliability and Bias in LLM-as-a-Judge Evaluation" (arXiv:2606.13685), which
measured pairwise judge verdicts flipping on average 13.6 percent of the time
across repeated identical evaluations.

``SelfConsistency`` exposes the same parameter names as the wrapped provider
method, so it plugs directly into ``Metric(implementation=...)`` with no
changes to ``Metric``, ``Selector``, or the evaluation pipeline, and it returns
``(score, {"reason": ...})`` so the per-trial breakdown flows into
``FeedbackCall.meta`` and is visible in OTEL spans and the dashboard.
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
import inspect
import logging
import math
import statistics
from typing import Any

logger = logging.getLogger(__name__)

_BUILTIN_STRATEGIES = frozenset({
    "mean",
    "median",
    "trimmed_mean",
    "majority_vote",
})


def _aggregate_scores(
    scores: list[float],
    aggregation: str | Callable[[list[float]], float],
    threshold: float,
) -> float:
    """Combine repeated-trial scores into a single value."""
    if callable(aggregation) and not isinstance(aggregation, str):
        return float(aggregation(scores))

    if aggregation == "mean":
        return statistics.mean(scores)

    if aggregation == "median":
        return statistics.median(scores)

    if aggregation == "trimmed_mean":
        if len(scores) < 3:
            return statistics.mean(scores)
        return statistics.mean(sorted(scores)[1:-1])

    if aggregation == "majority_vote":
        votes = sum(1 for s in scores if s >= threshold)
        if votes * 2 == len(scores):
            logger.warning(
                "SelfConsistency majority_vote tie (%d/%d). Falling back to "
                "median.",
                votes,
                len(scores),
            )
            return float(statistics.median(scores))
        return float(int(votes > len(scores) / 2))

    raise ValueError(f"Unknown aggregation: {aggregation!r}")


def _reliability_summary(
    scores: list[float], threshold: float
) -> dict[str, Any]:
    """Reliability signals over the repeated-trial scores.

    ``flip_rate`` and ``outcome_entropy`` are computed over the pass/fail view
    of the scores (each score binarised at ``threshold``), so they describe how
    often a pass/fail gate built on this judge would change on re-run.
    """
    n = len(scores)
    outcomes = [1 if s >= threshold else 0 for s in scores]
    n_pass = sum(outcomes)
    counts = [n - n_pass, n_pass]
    majority = max(counts)

    flip_rate = 1.0 - majority / n
    entropy = 0.0
    for c in counts:
        if c:
            p = c / n
            entropy -= p * math.log2(p)

    return {
        "n_trials": n,
        "scores": scores,
        "score_std": statistics.pstdev(scores) if n > 1 else 0.0,
        "flip_rate": flip_rate,
        "agreement": majority / n,
        "outcome_entropy": entropy,
    }


class SelfConsistency:
    """Run one LLM judge ``n_trials`` times and aggregate with a reliability
    summary.

    ``SelfConsistency`` wraps a single provider, calls the same named method on
    it ``n_trials`` times in parallel, aggregates the scores, and returns the
    aggregate together with a reliability summary. Because it exposes the same
    parameter names as the underlying provider method, it plugs directly into
    ``Metric(implementation=self_consistency)`` with no changes to ``Metric``,
    ``Selector``, or the evaluation pipeline.

    ``__call__`` always returns ``(score, {"reason": ..., "self_consistency":
    {...}})``, matching the ``_with_cot_reasons`` convention so the per-trial
    breakdown and reliability signals flow into ``FeedbackCall.meta`` and are
    visible in OTEL spans and the dashboard without any UI changes.

    Args:
        provider: An ``LLMProvider`` instance.
        method: Name of the feedback method to call, e.g. ``"relevance"`` or
            ``"groundedness_measure_with_cot_reasons"``.
        n_trials: Number of repeated calls. Must be at least 1.
        aggregation: How to combine trial scores. Accepts a strategy name
            (``"mean"``, ``"median"``, ``"trimmed_mean"``, ``"majority_vote"``)
            or any ``Callable[[list[float]], float]``. Defaults to ``"mean"``.
        threshold: Pass/fail boundary used for ``"majority_vote"`` and for the
            flip-rate and entropy view. Scores >= ``threshold`` count as a pass.
            Defaults to ``0.5``.
        max_workers: Maximum parallel threads. Defaults to ``n_trials``.

    Example::

        from trulens.core import Metric
        from trulens.feedback import SelfConsistency
        from trulens.providers.openai import OpenAI

        judge = SelfConsistency(
            provider=OpenAI(model_engine="gpt-4o-mini"),
            method="relevance",
            n_trials=5,
            aggregation="mean",
        )
        m = Metric(
            implementation=judge, name="Relevance (self-consistent)"
        ).on_input().on_output()
    """

    def __init__(
        self,
        provider: Any,
        method: str,
        n_trials: int = 5,
        aggregation: str | Callable[[list[float]], float] = "mean",
        *,
        threshold: float = 0.5,
        max_workers: int | None = None,
    ) -> None:
        if n_trials < 1:
            raise ValueError(f"n_trials must be >= 1, got {n_trials}.")

        if (
            isinstance(aggregation, str)
            and aggregation not in _BUILTIN_STRATEGIES
        ):
            raise ValueError(
                f"Unknown aggregation strategy {aggregation!r}. "
                f"Choose one of {sorted(_BUILTIN_STRATEGIES)} or pass a "
                "callable."
            )

        bound = getattr(provider, method, None)
        if bound is None or not callable(bound):
            raise AttributeError(
                f"Provider {type(provider).__name__!r} has no callable method "
                f"{method!r}."
            )

        self._provider = provider
        self._method = method
        self._n_trials = n_trials
        self._aggregation = aggregation
        self._threshold = threshold
        self._max_workers = max_workers or n_trials

        self.__signature__ = inspect.signature(bound)
        self.__name__ = f"self_consistency_{method}"

    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> tuple[float, dict[str, Any]]:
        """Evaluate the same arguments ``n_trials`` times and aggregate."""
        results: dict[int, tuple[float, str | None]] = {}

        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            future_to_idx = {
                executor.submit(self._call_once, args, dict(kwargs)): i
                for i in range(self._n_trials)
            }

            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    raw = future.result()
                    if isinstance(raw, tuple):
                        score = float(raw[0])
                        meta = (
                            raw[1]
                            if len(raw) > 1 and isinstance(raw[1], dict)
                            else {}
                        )
                        reason: str | None = meta.get("reason")
                    else:
                        score = float(raw)
                        reason = None
                    results[idx] = (score, reason)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "SelfConsistency trial %d failed: %s", idx, exc
                    )

        if not results:
            raise RuntimeError(
                f"All {self._n_trials} trials failed to produce a score."
            )

        ordered_idxs = sorted(results.keys())
        scores = [results[idx][0] for idx in ordered_idxs]
        agg_score = _aggregate_scores(
            scores, self._aggregation, self._threshold
        )
        reliability = _reliability_summary(scores, self._threshold)

        lines = [
            f"Aggregation: {self._aggregation} over {len(scores)} trials "
            f"-> {agg_score:.3f}",
            f"Flip rate: {reliability['flip_rate']:.3f} | "
            f"std: {reliability['score_std']:.3f} | "
            f"entropy: {reliability['outcome_entropy']:.3f}",
        ]
        for idx in ordered_idxs:
            score, reason = results[idx]
            lines.append(f"  trial {idx}: {score:.3f}")
            if reason:
                for line in reason.splitlines():
                    lines.append(f"    {line}")

        return agg_score, {
            "reason": "\n".join(lines),
            "self_consistency": reliability,
        }

    def _call_once(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        return getattr(self._provider, self._method)(*args, **kwargs)
