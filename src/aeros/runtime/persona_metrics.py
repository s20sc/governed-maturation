"""Operational metrics for the persona-adaptation surface (the state contract).

The metric set is fixed by the state contract and pinned by the
acceptance criterion ("each metric emits at least one sample under
representative test traffic"). The shape mirrors Prometheus / OTel
conventions so a future scraper can render them directly:

* ``persona_adaptation_events_total{kind, source}`` — counter; ``source``
  is ``"automatic"`` for emitter-issued tighten rows and ``"signed"``
  for operator CLI rows.
* ``persona_signature_gate_rejections_total{reason}`` — counter; bumped
  by the resolver gate (signature-policy / chain-verification) when an event is refused.
* ``persona_rate_limit_circuit_breaker_active`` — gauge (0/1); reflects
  the most recent governance evaluation per identity.
* ``persona_resolver_latency_seconds{quantile}`` — histogram exposing
  count, sum, and ``p50/p90/p99`` quantiles over the resolver fold.
* ``persona_adaptive_state{field}`` — gauge for the latest folded state
  (``risk_appetite_actual``, optional per-skill / per-action keys).
* ``persona_freeze_active`` — gauge (0/1) per identity.
* ``persona_floor_clamps_total{floor_kind}`` — counter for governance
  ``applied=False`` clamp activations (``floor`` / ``rate_limit`` /
  ``circuit_breaker``).

The registry is in-process and lock-protected; callers (resolver,
emitter, governance, route handler) flip counters without taking any
external dependency. Tests reach in via :func:`snapshot` /
:func:`reset_for_tests` to verify that
each metric emits at least one sample under representative traffic.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Final

# Quantiles surfaced by the resolver-latency histogram. The metric labels
# them as a histogram-with-quantile schema; for the in-process registry
# we approximate via a bounded reservoir + linear-interpolation summary.
_QUANTILES: Final[tuple[float, ...]] = (0.5, 0.9, 0.99)
# Bound the latency reservoir so a long-running bridge does not grow
# without limit. 4096 samples is enough for stable p99 estimates while
# keeping the in-process footprint small.
_LATENCY_RESERVOIR_CAP: Final[int] = 4096


@dataclass
class _Counter:
    """Label-keyed counter. Labels render in sorted-key order for stable
    output independent of insertion order."""

    name: str
    samples: dict[tuple[tuple[str, str], ...], float] = field(default_factory=dict)

    def inc(self, amount: float = 1.0, **labels: str) -> None:
        key = tuple(sorted(labels.items()))
        self.samples[key] = self.samples.get(key, 0.0) + amount

    def snapshot(self) -> list[dict[str, Any]]:
        return [
            {"labels": dict(k), "value": v}
            for k, v in sorted(self.samples.items())
        ]


@dataclass
class _Gauge:
    """Label-keyed gauge — last-write-wins."""

    name: str
    samples: dict[tuple[tuple[str, str], ...], float] = field(default_factory=dict)

    def set(self, value: float, **labels: str) -> None:
        key = tuple(sorted(labels.items()))
        self.samples[key] = float(value)

    def snapshot(self) -> list[dict[str, Any]]:
        return [
            {"labels": dict(k), "value": v}
            for k, v in sorted(self.samples.items())
        ]


@dataclass
class _Histogram:
    """Reservoir-backed summary surfacing count, sum, and quantiles.

    Fixed-size reservoir so a long-running process can't drift into
    unbounded memory; quantiles are linear-interpolated over the sorted
    reservoir which is sufficient for the the state contract dashboard view.
    """

    name: str
    count: int = 0
    sum: float = 0.0
    samples: list[float] = field(default_factory=list)

    def observe(self, value: float) -> None:
        self.count += 1
        self.sum += float(value)
        if len(self.samples) < _LATENCY_RESERVOIR_CAP:
            self.samples.append(float(value))
        else:
            # Replace a deterministic slot so the reservoir keeps a mix
            # of old + new samples without importing random in the hot
            # path. Slot index = (count - 1) % cap covers the reservoir
            # uniformly over time.
            self.samples[(self.count - 1) % _LATENCY_RESERVOIR_CAP] = float(value)

    def quantile(self, q: float) -> float:
        if not self.samples:
            return 0.0
        if not 0.0 <= q <= 1.0:
            raise ValueError(f"quantile must be in [0, 1], got {q}")
        ordered = sorted(self.samples)
        if len(ordered) == 1:
            return ordered[0]
        position = q * (len(ordered) - 1)
        lo = int(position)
        hi = min(lo + 1, len(ordered) - 1)
        frac = position - lo
        return ordered[lo] + (ordered[hi] - ordered[lo]) * frac

    def snapshot(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "sum": self.sum,
            "quantiles": {f"{q:.2f}": self.quantile(q) for q in _QUANTILES},
        }


class PersonaMetricsRegistry:
    """In-process Prometheus-shaped registry for the persona surface.

    Single global instance returned by :func:`get_registry`; tests call
    :func:`reset_for_tests` between cases to keep counters local.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._adaptation_events = _Counter("persona_adaptation_events_total")
        self._signature_rejections = _Counter("persona_signature_gate_rejections_total")
        self._floor_clamps = _Counter("persona_floor_clamps_total")
        self._circuit_breaker_active = _Gauge("persona_rate_limit_circuit_breaker_active")
        self._adaptive_state = _Gauge("persona_adaptive_state")
        self._freeze_active = _Gauge("persona_freeze_active")
        self._resolver_latency = _Histogram("persona_resolver_latency_seconds")

    # -------- counters / gauges (called from emit, resolver, governance) -------- #

    def inc_adaptation_event(self, kind: str, source: str) -> None:
        with self._lock:
            self._adaptation_events.inc(kind=kind, source=source)

    def inc_signature_rejection(self, reason: str) -> None:
        with self._lock:
            self._signature_rejections.inc(reason=reason)

    def inc_floor_clamp(self, floor_kind: str) -> None:
        with self._lock:
            self._floor_clamps.inc(floor_kind=floor_kind)

    def set_circuit_breaker_active(
        self, active: bool, identity_hash: str = "default"
    ) -> None:
        with self._lock:
            self._circuit_breaker_active.set(
                1.0 if active else 0.0, identity=identity_hash
            )

    def set_freeze_active(self, frozen: bool, identity_hash: str = "default") -> None:
        with self._lock:
            self._freeze_active.set(
                1.0 if frozen else 0.0, identity=identity_hash
            )

    def set_adaptive_state(self, field: str, value: float) -> None:
        with self._lock:
            self._adaptive_state.set(float(value), field=field)

    def observe_resolver_latency(self, seconds: float) -> None:
        with self._lock:
            self._resolver_latency.observe(seconds)

    # ---------------------------- snapshot ---------------------------- #

    def snapshot(self) -> dict[str, Any]:
        """Return a JSON-safe view of every metric for the HTTP endpoint."""
        with self._lock:
            return {
                "persona_adaptation_events_total": self._adaptation_events.snapshot(),
                "persona_signature_gate_rejections_total": (
                    self._signature_rejections.snapshot()
                ),
                "persona_floor_clamps_total": self._floor_clamps.snapshot(),
                "persona_rate_limit_circuit_breaker_active": (
                    self._circuit_breaker_active.snapshot()
                ),
                "persona_adaptive_state": self._adaptive_state.snapshot(),
                "persona_freeze_active": self._freeze_active.snapshot(),
                "persona_resolver_latency_seconds": self._resolver_latency.snapshot(),
            }

    def reset(self) -> None:
        with self._lock:
            self._adaptation_events.samples.clear()
            self._signature_rejections.samples.clear()
            self._floor_clamps.samples.clear()
            self._circuit_breaker_active.samples.clear()
            self._adaptive_state.samples.clear()
            self._freeze_active.samples.clear()
            self._resolver_latency.count = 0
            self._resolver_latency.sum = 0.0
            self._resolver_latency.samples.clear()


_REGISTRY = PersonaMetricsRegistry()
_REGISTRY_LOCK = threading.Lock()


def get_registry() -> PersonaMetricsRegistry:
    """Return the process-global registry. The bridge / CLI / tests share it."""
    return _REGISTRY


def reset_for_tests() -> None:
    """Drop every recorded sample. Tests call this between cases so a
    counter incremented in one test doesn't leak into the next."""
    with _REGISTRY_LOCK:
        _REGISTRY.reset()


@contextmanager
def time_resolver_fold() -> Iterator[None]:
    """Context manager wrapping a resolver fold; observes wall-time into
    ``persona_resolver_latency_seconds``. The resolver picks this up via
    a thin wrapper so the pure-fold path stays IO-free."""
    t0 = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - t0
        _REGISTRY.observe_resolver_latency(elapsed)


def record_adaptive_state(adaptive: Any) -> None:
    """Push the current ``PersonaAdaptive`` overlay into the gauge.

    Surfaces ``risk_appetite_actual`` plus per-skill confidence and per-
    action-kind threshold entries so dashboards can chart drift over
    time. ``adaptive`` is duck-typed (``PersonaAdaptive``) — pulling the
    real type would create a runtime import cycle with persona_resolver.
    """
    risk = getattr(adaptive, "risk_appetite_actual", None)
    if risk is not None:
        _REGISTRY.set_adaptive_state("risk_appetite_actual", float(risk))
    skill = getattr(adaptive, "skill_confidence", {}) or {}
    for cap, value in skill.items():
        _REGISTRY.set_adaptive_state(f"skill_confidence:{cap}", float(value))
    threshold = getattr(adaptive, "human_check_threshold", {}) or {}
    for action_kind, value in threshold.items():
        _REGISTRY.set_adaptive_state(
            f"human_check_threshold:{action_kind}", float(value)
        )


__all__ = [
    "PersonaMetricsRegistry",
    "get_registry",
    "record_adaptive_state",
    "reset_for_tests",
    "time_resolver_fold",
]
