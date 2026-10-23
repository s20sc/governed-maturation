"""Exercise real E5 computation using only a copied artifact."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def test_e5_runs_from_artifact_without_parent_audit(tmp_path: Path) -> None:
    source = Path(__file__).resolve().parents[1]
    copied = tmp_path / "artifact"
    shutil.copytree(source, copied, ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache", "out", ".venv"))
    assert not (copied.parent / "audit").exists()
    output = tmp_path / "diagnostic.json"
    program = """
import importlib.util
import json
from pathlib import Path
import sys
script = Path('scripts/persona_experiment_e5_matched_controls.py').resolve()
spec = importlib.util.spec_from_file_location('portable_e5', script)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
result = module.benchmark(seeds=(4200,), repetitions=100)
# Editable installers create build metadata; it is not experiment source.
generated = Path('src/governed_maturation.egg-info')
generated.mkdir(exist_ok=True)
(generated / 'SOURCES.txt').write_text('machine-specific editable build file list')
after_install_metadata = module.benchmark(seeds=(4200,), repetitions=100)
assert after_install_metadata == result
Path(sys.argv[1]).write_text(json.dumps(result, allow_nan=False))
assert Path(module.effective_persona.__code__.co_filename).resolve().is_relative_to(Path.cwd())
"""
    environment = dict(os.environ, PYTHONPATH=str(copied / "src"), PYTHONDONTWRITEBYTECODE="1")
    completed = subprocess.run(
        [sys.executable, "-c", program, str(output)], cwd=copied,
        env=environment, capture_output=True, text=True, timeout=120,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    result = json.loads(output.read_text())
    assert result["seeds"] == [4200]
    assert result["bootstrap_repetitions"] == 100
    assert len(result["per_seed"]) == 140
    assert len(result["paired_contrasts"]) == 120
    assert all(result["signature_probe"].values())
    definitions = result["metric_definitions"]
    assert len(definitions) == 10
    assert set(definitions) == set(result["paired_contrasts"][0]["metrics"])
    assert len(set(definitions.values())) == 10
    assert all(len(value) > 30 and "See " not in value for value in definitions.values())
    expected = {
        *(path.relative_to(copied).as_posix() for path in (copied / "src").rglob("*")
          if path.is_file() and path.suffix in (".py", ".txt")
          and not any(part.endswith(".egg-info") for part in path.relative_to(copied).parts)),
        *(path.relative_to(copied).as_posix() for path in (copied / "scripts").glob("*.py")),
        *(path.relative_to(copied).as_posix() for path in (copied / "tests").glob("*.py")),
        "reproduce.py", "EXPERIMENT_PROTOCOL.md", "METRICS.md", "pyproject.toml", "requirements-tested.txt",
    }
    assert set(result["provenance_sha256"]) == expected
    for relative, digest in result["provenance_sha256"].items():
        path = Path(relative)
        assert not path.is_absolute() and ".." not in path.parts
        target = (copied / path).resolve()
        assert target.is_relative_to(copied.resolve())
        assert hashlib.sha256(target.read_bytes()).hexdigest() == digest
    # Actual protected trajectories must survive the standalone run.
    for row in result["per_seed"]:
        if row["profile"] != "all":
            continue
        if row["scenario"] == "T1_success_only":
            assert row["authorization_rejected"] == row["attempts"]
            assert row["final_confidence"] == row["initial_confidence"]
        elif row["scenario"] == "T5_hallucination":
            assert row["validation_rejected"] == row["attempts"]
        elif row["scenario"] == "T4_failure_flood":
            assert row["first_circuit_attempt"] == 101
