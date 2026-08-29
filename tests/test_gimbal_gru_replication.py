import json
from dataclasses import replace

import pytest

pytest.importorskip("torch")

from autonomous_observation_lab.gimbal_servoing import (
    GimbalDatasetGenerationConfig,
    GimbalDomainRandomizationConfig,
    ObservationProfile,
    generate_gimbal_dataset,
    save_gimbal_dataset,
)
from autonomous_observation_lab.gimbal_servoing.closed_loop import (
    nominal_scenario,
)
from autonomous_observation_lab.gimbal_servoing.gru_replication import (
    GRUReplicationConfig,
    run_gru_o2_replication,
)
from autonomous_observation_lab.gimbal_servoing.visualization import (
    _gru_replication_markdown,
    _load_gru_replication_result,
)


def test_o2_replication_keeps_training_seeds_as_the_replication_unit(tmp_path):
    base = nominal_scenario()
    scenario = replace(
        base,
        name="replication_smoke",
        config=replace(
            base.config,
            timing=replace(base.config.timing, episode_duration_s=0.2),
        ),
    )
    randomization = GimbalDomainRandomizationConfig()
    randomization = replace(
        randomization,
        hardware=replace(
            randomization.hardware,
            episode_duration_s=0.2,
        ),
    )
    paths = {}
    for split, seed in (("train", 2101), ("validation", 2201), ("test", 2301)):
        dataset = generate_gimbal_dataset(
            GimbalDatasetGenerationConfig(
                split=split,
                seeds=(seed,),
                scenario_names=(scenario.name,),
                behavior_names=(
                    "proportional_rate",
                    "proportional_position",
                ),
                observation_profiles=(
                    ObservationProfile.DISTURBANCE_AWARE,
                ),
                prediction_horizons_s=(0.0,),
                domain_randomization=randomization,
                include_oracle_ceilings=False,
            ),
            scenarios=(scenario,),
        )
        paths[split], _ = save_gimbal_dataset(tmp_path / split, dataset)

    result = run_gru_o2_replication(
        train_path=paths["train"],
        validation_path=paths["validation"],
        test_path=paths["test"],
        checkpoint_directory=tmp_path / "checkpoints",
        config=GRUReplicationConfig(
            training_seeds=(5, 7),
            epochs=1,
            batch_size=1,
            hidden_dim=8,
            embedding_dim=8,
        ),
    )

    assert result["validation_variant_count"] == 1
    assert result["test_variant_count"] == 1
    assert len(result["training_seed_results"]) == 2
    assert {
        seed_result["training_seed"]
        for seed_result in result["training_seed_results"]
    } == {5, 7}
    assert set(result["replication_summary"]) == {"rate", "position"}
    for mode in ("rate", "position"):
        summary = result["replication_summary"][mode]
        assert summary["training_seed_count"] == 2
        assert summary["selected_horizon_consistent"]
        assert summary["learned_metric_distribution"][
            "mean_control_cost"
        ]["sample_std"] >= 0.0
    assert all(
        (tmp_path / "checkpoints" / f"gimbal_gru_o2_seed_{seed}.pt").exists()
        for seed in (5, 7)
    )

    artifact = tmp_path / "replication.json"
    artifact.write_text(json.dumps(result), encoding="utf-8")
    loaded = _load_gru_replication_result(artifact)
    markdown = _gru_replication_markdown(loaded)
    assert "2 independently initialized O2 models" in markdown
    assert "Training seed—not test episode—is the replication unit" in markdown
    assert "## Rate control" in markdown
    assert "## Position control" in markdown
