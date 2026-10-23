"""Behavioral controls for the independent E5 synthetic benchmark."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/persona_experiment_e5_matched_controls.py"
spec = importlib.util.spec_from_file_location("e5_matched", SCRIPT)
e5 = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = e5
spec.loader.exec_module(e5)


@pytest.mark.parametrize("mode", ["discrete", "ema"])
def test_t1_is_success_only_and_real_resolver_rejects(mode: str) -> None:
    assert all(item.success for item in e5.schedule(4200, "T1_success_only"))
    full = e5.run_arm(4200, "T1_success_only", mode)
    bypass = e5.run_arm(4200, "T1_success_only", mode, "no_authorization")
    assert full["authorization_rejected"] == full["attempts"]
    assert full["final_confidence"] == pytest.approx(full["initial_confidence"])
    assert bypass["final_confidence"] > full["final_confidence"]
    assert bypass["rate_limit"] > 0  # only authorization was bypassed


def test_real_ed25519_negative_and_positive_controls() -> None:
    assert all(e5.signature_probe().values())


@pytest.mark.parametrize("mode", ["discrete", "ema"])
def test_actual_governance_circuit_and_independent_rate_ablation(mode: str) -> None:
    full = e5.run_arm(4200, "T4_failure_flood", mode)
    no_circuit = e5.run_arm(4200, "T4_failure_flood", mode, "no_circuit")
    no_rate = e5.run_arm(4200, "T4_failure_flood", mode, "no_rate")
    assert full["first_circuit_attempt"] == 101
    assert full["circuit_breaker"] == full["attempts"] - 100
    assert full["rate_limit"] == 50
    assert no_circuit["circuit_breaker"] == 0
    assert no_circuit["rate_limit"] == no_circuit["attempts"] - 50
    assert no_rate["applied"] == 100
    assert no_rate["circuit_breaker"] == full["circuit_breaker"]


@pytest.mark.parametrize("mode", ["discrete", "ema"])
def test_actual_validation_rejects_unobserved_skill_and_ablation_exposes_it(mode: str) -> None:
    full = e5.run_arm(4200, "T5_hallucination", mode)
    bypass = e5.run_arm(4200, "T5_hallucination", mode, "no_validation")
    assert full["validation_rejected"] == full["attempts"]
    assert full["governance_rows"] == 0
    assert bypass["applied"] == bypass["attempts"]
    assert bypass["total_variation"] > 0


@pytest.mark.parametrize("mode", ["discrete", "ema"])
@pytest.mark.parametrize("floor", [.05, .1, .2])
def test_floor_sweep_respects_selected_bound(mode: str, floor: float) -> None:
    full = e5.run_arm(4200, "floor_pressure", mode, floor=floor)
    bypass = e5.run_arm(4200, "floor_pressure", mode, "no_floor", floor=floor)
    assert full["floor"] > 0
    assert full["minimum_confidence"] >= floor
    assert bypass["minimum_confidence"] < floor
    assert full["rate_limit"] == full["circuit_breaker"] == 0


@pytest.mark.parametrize("mode", ["discrete", "ema"])
@pytest.mark.parametrize("ceiling", [.9, .95, .99])
def test_harness_ceiling_is_explicit_and_effective(mode: str, ceiling: float) -> None:
    full = e5.run_arm(4200, "ceiling_pressure", mode, ceiling=ceiling)
    bypass = e5.run_arm(4200, "ceiling_pressure", mode, "no_floor", ceiling=ceiling)
    assert full["final_human_threshold"] <= ceiling
    assert bypass["final_human_threshold"] > full["final_human_threshold"]
    assert full["floor"] > 0


def test_pairing_reproducibility_and_ci() -> None:
    a = e5.run_arm(4200, "floor_pressure", "discrete")
    b = e5.run_arm(4200, "floor_pressure", "ema", "no_floor")
    assert a["evidence_sha256"] == b["evidence_sha256"]
    assert a == e5.run_arm(4200, "floor_pressure", "discrete")
    assert e5.paired_ci([0.] * 30, 100)["ci95_percentile"] == [0., 0.]
    assert e5.paired_ci([1.] * 30, 100)["mean_difference"] == 1.
    with pytest.raises(ValueError):
        e5.benchmark((1, 2), 100)


def test_default_output_does_not_target_data_and_existing_output_is_protected(tmp_path: Path) -> None:
    target = tmp_path / "existing.json"
    target.write_text("preserve")
    with pytest.raises(SystemExit):
        e5.main(["--output", str(target)])
    assert target.read_text() == "preserve"


def test_validation_positive_control_and_production_ceiling_gap() -> None:
    proposal = e5.ProposedDelta(target_kind="skill_confidence_decrease",
        target_capability=e5.CAP, proposed_delta={"magnitude": .1}, confidence=.9)
    assert not e5.EMAValidationBackend().validate([proposal], capability_failure_rates={e5.CAP: 1.})[0].rejected
    ev = e5.event("human_check_threshold_tighten", .1, e5.NOW)
    adaptive = e5.effective_persona(e5.PersonaCore(), []).adaptive
    from dataclasses import replace
    adaptive = replace(adaptive, human_check_threshold={"synthetic.action": .94})
    assert not e5.governance.would_violate_floor(adaptive, ev, .7)
    _, reason = e5.decide(ev, [], adaptive, "all", .05, .95)
    assert reason == "floor"
