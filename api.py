"""Lightweight API shim used by the Streamlit frontend during development.

Provides `run_optimization` and `explain_source` so the app can run even when
backend services or trained models are not available.
"""
from typing import Any, Dict, List
import random
import numpy as np

__all__ = ["run_optimization", "explain_source"]


def _make_assignment(source_id: str, risk: float):
    return {"source_id": source_id, "task": "Task_A", "expected_risk": float(round(risk, 2))}


def run_optimization(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Return a minimal, deterministic optimization response for UI development.

    payload: {"sources": [ {"source_id":..., "features": {...}, ... }, ... ], "seed": int}
    """
    sources = payload.get("sources", [])
    seed = payload.get("seed", 42)
    random.seed(seed)

    ml_policy = []
    det_policy = []
    uni_policy = []

    for s in sources:
        sid = s.get("source_id", "unknown")
        # deterministic: fixed moderate risk
        det_policy.append(_make_assignment(sid, 0.5))
        # uniform: equal distribution -> moderate risk as well
        uni_policy.append(_make_assignment(sid, 0.5))
        # ml: use a pseudo-random score to emulate nontrivial decisions
        ml_policy.append(_make_assignment(sid, random.random()))

    def _sum_risk(policy):
        return round(sum(a["expected_risk"] for a in policy), 2)

    emv = {
        "ml_tssp": _sum_risk(ml_policy),
        "deterministic": _sum_risk(det_policy),
        "uniform": _sum_risk(uni_policy)
    }

    # Source-level EVPI placeholder: zeros
    source_evpi = {s.get("source_id", "unknown"): 0.0 for s in sources}

    return {
        "policies": {
            "ml_tssp": ml_policy,
            "deterministic": det_policy,
            "uniform": uni_policy
        },
        "emv": emv,
        "evpi": round(max(emv.values()) - min(emv.values()), 2),
        "source_evpi": source_evpi,
        "audit_log": {"note": "This is a development stub of run_optimization."}
    }


def explain_source(source: Any) -> Dict[str, Any]:
    """Return a SHAP-style explanation if available, otherwise a user-friendly message.

    Accepts either a features dict or the full source dict (with a "features" key).
    """
    # Extract features dict if the full source object was provided
    if isinstance(source, dict) and "features" in source:
        features = source.get("features")
    else:
        features = source

    try:
        # Defer importing dashboard until runtime to avoid import cycles
        import dashboard
        # dashboard.explain_source returns a mapping: {class: {feature: value, ...}}
        shap_map = dashboard.explain_source(features)
        return {"model": getattr(dashboard, "MODEL_VERSION", "unknown"), "shap_values": shap_map}
    except RuntimeError as re:
        # SHAP not available (no model/explainer); return an explicit response
        return {"message": str(re), "shap_values": {}}
    except Exception as e:
        # Unexpected error; return a safe fallback
        return {"message": "Explanation unavailable: %s" % str(e), "shap_values": {}}
