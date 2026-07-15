"""
Self-test-only fixtures (synthetic schedules/bundles/transports/
tokenizers), mechanically moved out of run_phase_a.py's former
"Self-test" section, verbatim -- no logic changes.
"""

from __future__ import annotations

from run_phase_a import *  # noqa: F401,F403
from run_phase_a import _check_model_structure  # noqa: F401


def _build_fixture_episodes(model: str, seed: int) -> list[Episode]:
    """
    Self-test-only fixture: builds a small, internally-consistent
    1-repeat (2-block, 8-episode) synthetic schedule for a single model,
    used purely to exercise `_check_model_structure()`'s logic in
    isolation. This is NOT the frozen generator, is NOT used by any real
    validation path, and does not claim to match the official 80-episode
    contract (which needs 5 repeats x 2 models).
    """
    episodes: list[Episode] = []
    states = [(0, "low"), (12, "high")]
    cells = [
        (concurrency, condition)
        for concurrency in OFFICIAL_CONCURRENCIES
        for condition in OFFICIAL_CONDITIONS
    ]
    block_number = 0
    for offload_gb, state_label in states:
        block_number += 1
        block_id = f"{model}_block{block_number:02d}_{state_label}"
        for order_in_block, (concurrency, condition) in enumerate(cells, start=1):
            repeat = 1
            episode_id = (
                f"{model}_off{offload_gb}_conc{concurrency}_{condition}_"
                f"rep{repeat}"
            )
            episodes.append(
                Episode(
                    episode_id=episode_id,
                    model=model,
                    offload_gb=offload_gb,
                    state_label=state_label,
                    concurrency=concurrency,
                    condition=condition,
                    repeat=repeat,
                    random_seed=seed,
                    episode_seed=derive_seed(str(seed), episode_id),
                    victim_workload_seed=derive_seed(
                        str(seed), model, str(concurrency), str(repeat)
                    ),
                    burst_workload_seed=derive_seed(
                        str(seed), model, str(concurrency), str(repeat), "burst"
                    ),
                    victim_request_count=20,
                    victim_input_len=256,
                    victim_output_len=64,
                    victim_temperature=0.0,
                    burst_parallel_requests=4,
                    burst_input_len=256,
                    burst_output_len=256,
                    burst_temperature=0.0,
                    restart_server_before_block=1 if order_in_block == 1 else 0,
                    block_id=block_id,
                    order_in_block=order_in_block,
                )
            )
    return episodes


def _make_fixture_block_bundle(model: str, seed: int) -> tuple["LoadedBundle", str]:
    episodes = _build_fixture_episodes(model, seed)
    block_id = episodes[0].block_id
    bundle = LoadedBundle(
        schedule_dir=Path("/nonexistent-fixture-only"),
        json_obj={"seed": seed},
        csv_fieldnames=[],
        csv_rows=[],
        audit_text="",
        episodes=episodes,
        fingerprint="sha256:" + "a" * 64,
    )
    return bundle, block_id


def _success_script_factory(payload: dict) -> "FakeStreamScript":
    return FakeStreamScript(
        prompt_token_ids_echo=list(payload["prompt"]),
        token_events=[[9000 + k] for k in range(payload["max_tokens"])],
        usage={"prompt_tokens": len(payload["prompt"]), "completion_tokens": payload["max_tokens"]},
    )


def _make_success_transport() -> "FakeTransport":
    t = FakeTransport()
    t.default_script_factory = _success_script_factory
    t.set_get_response(HEALTH_ENDPOINT, 200, {})
    t.set_get_response(
        MODELS_ENDPOINT, 200,
        {"data": [{"id": MODEL_FULL_ID["llama"]}, {"id": MODEL_FULL_ID["qwen"]}]},
    )
    t.set_get_response(OPENAPI_ENDPOINT, 200, {"paths": {COMPLETIONS_ENDPOINT: {}}})
    return t


def _make_fixture_campaign_bundle(seed: int) -> "LoadedBundle":
    """Small synthetic multi-block, multi-model bundle for Stage-3 fake-
    campaign tests: 2 models x 2 states = 4 blocks x 4 episodes = 16
    episodes. Uses the same official 256/64/256 request dimensions as
    the real schedule, just far fewer blocks."""
    episodes = _build_fixture_episodes("llama", seed) + _build_fixture_episodes("qwen", seed)
    return LoadedBundle(
        schedule_dir=Path("/nonexistent-fixture-campaign-only"),
        json_obj={"seed": seed, "design_version": "fixture-campaign-v1"},
        csv_fieldnames=[],
        csv_rows=[],
        audit_text="",
        episodes=episodes,
        fingerprint="sha256:" + "c" * 64,
    )


def _fixture_tokenizer_factory(model_key: str) -> "FakeTokenizerAdapter":
    return FakeTokenizerAdapter(vocab_size=2000, special_token_ids={0, 1, 2})


