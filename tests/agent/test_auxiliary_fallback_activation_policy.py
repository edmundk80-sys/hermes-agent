"""Runtime policy tests for top-level fallbacks used by auxiliary calls."""

from unittest.mock import MagicMock, patch

from agent.auxiliary_client import (
    _try_configured_fallback_chain,
    _try_main_fallback_chain,
)


_CHAIN = [
    {"provider": "openai-codex", "model": "gpt-5.6-sol"},
    {"provider": "xai", "model": "grok-4.5"},
]


def _exercise_main_chain():
    resolved = []

    def resolve_entry(entry):
        resolved.append(entry["provider"])
        return MagicMock(name=entry["provider"]), entry["model"]

    with (
        patch(
            "hermes_cli.config.load_config_readonly",
            return_value={"fallback_providers": _CHAIN},
        ),
        patch("agent.auxiliary_client._read_main_provider", return_value="anthropic"),
        patch("agent.auxiliary_client._is_provider_unhealthy", return_value=False),
        patch(
            "agent.auxiliary_client._task_minimum_context_length",
            return_value=None,
        ),
        patch(
            "agent.auxiliary_client._resolve_fallback_entry",
            side_effect=resolve_entry,
        ),
    ):
        result = _try_main_fallback_chain(
            "compression", "anthropic", "unavailable"
        )
    return result, resolved


def test_parent_auxiliary_main_chain_skips_openai_codex():
    """Parent auxiliary recovery must continue past delegation-only Codex."""
    (client, model, provider), resolved = _exercise_main_chain()

    assert client is not None
    assert model == "grok-4.5"
    assert provider == "xai"
    assert resolved == ["xai"]


def test_delegated_child_auxiliary_main_chain_preserves_openai_codex():
    """A delegated child keeps the configured Codex/Sol fallback route."""
    from agent.delegation_context import delegated_child_context

    with delegated_child_context():
        (client, model, provider), resolved = _exercise_main_chain()

    assert client is not None
    assert model == "gpt-5.6-sol"
    assert provider == "openai-codex"
    assert resolved == ["openai-codex"]


def test_parent_auxiliary_task_chain_skips_openai_codex():
    """Task-specific fallback chains obey the same parent activation policy."""
    resolved = []

    def resolve_entry(entry):
        resolved.append(entry["provider"])
        return MagicMock(name=entry["provider"]), entry["model"]

    with (
        patch(
            "agent.auxiliary_client._get_auxiliary_task_config",
            return_value={"fallback_chain": _CHAIN},
        ),
        patch(
            "agent.auxiliary_client._task_minimum_context_length",
            return_value=None,
        ),
        patch(
            "agent.auxiliary_client._resolve_fallback_entry",
            side_effect=resolve_entry,
        ),
    ):
        client, model, label = _try_configured_fallback_chain(
            "compression", "anthropic", "unavailable"
        )

    assert client is not None
    assert model == "grok-4.5"
    assert label == "fallback_chain[1](xai)"
    assert resolved == ["xai"]
