from __future__ import annotations

from pathlib import Path

from tinyquant.orchestration.h4_cycle import run_h4_cycle


def test_h4_cycle_synthetic_writes_audit(strategy_config) -> None:
    out = run_h4_cycle(strategy_config, equity_usd=10_000.0, synthetic=True)
    assert out.get("skipped") is not True
    assert "gat_scores" in out
    assert "sentiment" in out
    assert "per_tradable" in out["sentiment"]
    audit_dir = Path(strategy_config.observability.audit_dir)
    audit_files = list(audit_dir.glob("h4_cycle_*.json"))
    assert len(audit_files) == 1
