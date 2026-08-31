import json

import pytest

from autonomous_observation_lab.gimbal_servoing.predictive_position_v31 import (
    default_v31_candidates,
    evaluate_predictive_position_v31,
)


def test_v31_candidates_require_joint_risk_and_are_predeclared():
    candidates = default_v31_candidates()

    assert len(candidates) == 4
    assert len({candidate.name for candidate in candidates}) == 4
    assert all(
        candidate.controller.activation_gate_mode in {"minimum", "product"}
        for candidate in candidates
    )


def test_v31_rejects_v3_seed_reuse_before_loading_v21(tmp_path):
    predecessor = tmp_path / "v3.json"
    predecessor.write_text(
        json.dumps(
            {
                "experiment": (
                    "gimbal_constrained_predictive_position_v3_protocol_v1"
                ),
                "confirmation": {
                    "opened": True,
                    "recommendation": "retain_visibility_risk_v21",
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="historical evaluation seeds"):
        evaluate_predictive_position_v31(
            v3_results=predecessor,
            visibility_risk_results=tmp_path / "missing.json",
            development_seeds=(84000,),
            confirmation_seeds=(86000,),
        )
