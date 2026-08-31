import math
from dataclasses import replace

from autonomous_observation_lab.gimbal_servoing import (
    BeliefRecoveryConfig,
    BeliefRecoveryController,
    ConstantVelocityEstimatorConfig,
    ConstantVelocityTargetEstimator,
    GimbalCommandMode,
    ObservationProfile,
    RecoveryState,
    StaticAngularMotion,
    TargetStateRateController,
    recovery_scenarios,
)
from autonomous_observation_lab.gimbal_servoing.closed_loop import (
    run_closed_loop_controller,
)


def test_belief_recovery_traverses_declared_states_after_detector_burst():
    scenario = recovery_scenarios()[0]
    config = replace(
        scenario.config,
        command_mode=GimbalCommandMode.RATE,
        observation_profile=ObservationProfile.DISTURBANCE_AWARE,
    )
    estimator = ConstantVelocityTargetEstimator(
        ConstantVelocityEstimatorConfig(
            selected_axis_fov_rad=config.camera.selected_axis_fov_rad,
            center_noise_std_normalized=(
                config.camera.center_noise_std_normalized
            ),
            max_prediction_horizon_s=0.30,
            history_horizon_s=1.0,
        )
    )
    delegate = TargetStateRateController(
        estimator=estimator,
        max_rate_rad_s=config.servo.max_rate_rad_s,
        proportional_gain_s_inv=2.5,
    )
    controller = BeliefRecoveryController(
        delegate=delegate,
        servo=config.servo,
        command_mode=GimbalCommandMode.RATE,
    )
    run = run_closed_loop_controller(
        name="belief",
        description="test",
        scenario=scenario,
        config=config,
        controller=controller,
        seed=7,
    )

    transitions = [transition.current for transition in controller.transitions]
    assert transitions[:4] == [
        RecoveryState.COAST,
        RecoveryState.SEARCH,
        RecoveryState.REACQUIRE,
        RecoveryState.TRACK,
    ]
    assert controller.transitions[0].time_s > 2.0
    assert run.metrics.unrecovered_loss_events == 0
    assert len(controller.action_trace) == len(controller.state_trace)

    action_by_time = dict(controller.action_trace)
    for transition in controller.transitions:
        if transition.current is not RecoveryState.REACQUIRE:
            continue
        index = next(
            index
            for index, (time_s, _) in enumerate(controller.action_trace)
            if math.isclose(time_s, transition.time_s)
        )
        assert action_by_time[transition.time_s] == controller.action_trace[
            index - 1
        ][1]


def test_recovery_suite_separates_reentry_from_physical_impossibility():
    scenarios = {scenario.name: scenario for scenario in recovery_scenarios()}
    returning = scenarios["travel_limit_reentry"].target_motion
    unreachable = scenarios["physically_unreachable"].target_motion

    assert math.degrees(returning.state_at(3.0)[0]) > 80.0
    assert returning.state_at(7.0)[0] == 0.0
    assert math.degrees(unreachable.state_at(7.0)[0]) > 80.0


def test_edge_conditioning_holds_centered_dropout_but_searches_boundary_exit():
    scenarios = {scenario.name: scenario for scenario in recovery_scenarios()}
    centered_base = scenarios["detector_burst_recovery"]
    centered = replace(
        centered_base,
        name="centered_detector_outage",
        target_motion=StaticAngularMotion(),
        body_motion=StaticAngularMotion(),
        config=replace(
            centered_base.config,
            camera=replace(
                centered_base.config.camera,
                forced_dropout_intervals_s=((1.0, 2.0),),
            ),
            timing=replace(
                centered_base.config.timing,
                episode_duration_s=3.0,
            ),
        ),
    )
    recovery = BeliefRecoveryConfig(
        maximum_coast_s=0.35,
        maximum_coast_bearing_std_rad=math.radians(10.0),
        edge_conditioned_search=True,
        search_activation_edge_fraction=0.60,
        search_activation_minimum_outward_rate_normalized_s=0.0,
    )

    def rollout(scenario):
        config = replace(
            scenario.config,
            command_mode=GimbalCommandMode.RATE,
            observation_profile=ObservationProfile.DISTURBANCE_AWARE,
        )
        estimator = ConstantVelocityTargetEstimator(
            ConstantVelocityEstimatorConfig(
                selected_axis_fov_rad=config.camera.selected_axis_fov_rad,
                center_noise_std_normalized=(
                    config.camera.center_noise_std_normalized
                ),
                max_prediction_horizon_s=0.30,
                history_horizon_s=1.0,
            )
        )
        delegate = TargetStateRateController(
            estimator=estimator,
            max_rate_rad_s=config.servo.max_rate_rad_s,
            proportional_gain_s_inv=2.5,
        )
        controller = BeliefRecoveryController(
            delegate=delegate,
            servo=config.servo,
            command_mode=GimbalCommandMode.RATE,
            recovery=recovery,
        )
        run_closed_loop_controller(
            name="edge",
            description="test",
            scenario=scenario,
            config=config,
            controller=controller,
            seed=7,
        )
        return controller

    centered_controller = rollout(centered)
    boundary_controller = rollout(scenarios["travel_limit_reentry"])

    assert RecoveryState.COAST in {
        transition.current for transition in centered_controller.transitions
    }
    assert RecoveryState.SEARCH not in {
        transition.current for transition in centered_controller.transitions
    }
    assert RecoveryState.SEARCH in {
        transition.current for transition in boundary_controller.transitions
    }
    assert any(
        transition.reason == "coast limit with edge evidence"
        for transition in boundary_controller.transitions
    )
    assert not any(
        supported for _, supported in centered_controller.edge_evidence_trace
    )
    assert any(
        supported for _, supported in boundary_controller.edge_evidence_trace
    )
