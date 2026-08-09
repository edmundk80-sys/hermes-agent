"""Test that AuthError triggers fallback provider resolution (#7230)."""

from unittest.mock import patch

import pytest


class TestResolveRuntimeAgentKwargsAuthFallback:
    """_resolve_runtime_agent_kwargs should try fallback on AuthError."""

    def test_auth_error_tries_fallback(self, tmp_path, monkeypatch):
        """When primary provider raises AuthError, fallback is attempted."""
        from hermes_cli.auth import AuthError

        # Create a config with fallback
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            "model:\n  provider: openai-codex\n"
            "fallback_model:\n  provider: openrouter\n"
            "  model: meta-llama/llama-4-maverick\n"
        )

        monkeypatch.setattr("gateway.run._hermes_home", tmp_path)

        call_count = {"n": 0}

        def _mock_resolve(**kwargs):
            call_count["n"] += 1
            # First call = primary path (gateway reads model.provider from
            # config.yaml internally; we simulate the auth failure here).
            # Second call = fallback path with explicit_api_key + explicit_base_url
            # supplied by gateway from fallback_model config.
            if call_count["n"] == 1:
                raise AuthError("Codex token refresh failed with status 401")
            return {
                "api_key": "fallback-key",
                "base_url": "https://openrouter.ai/api/v1",
                "provider": "openrouter",
                "api_mode": "openai_chat",
                "command": None,
                "args": None,
                "credential_pool": None,
            }

        with patch(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            side_effect=_mock_resolve,
        ):
            from gateway.run import _resolve_runtime_agent_kwargs
            result = _resolve_runtime_agent_kwargs()

        assert result["provider"] == "openrouter"
        assert result["api_key"] == "fallback-key"
        # Should have been called at least twice (primary + fallback)
        assert call_count["n"] >= 2

    def test_parent_startup_skips_openai_codex_fallback(self, tmp_path, monkeypatch):
        """A stale gateway chain must not activate delegation-only Codex."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            "fallback_providers:\n"
            "  - provider: openai-codex\n"
            "    model: gpt-5.6-sol\n"
            "  - provider: openrouter\n"
            "    model: x-ai/grok-4.5\n",
            encoding="utf-8",
        )
        monkeypatch.setattr("gateway.run._hermes_home", tmp_path)
        requested = []

        def _mock_resolve(**kwargs):
            provider = kwargs.get("requested")
            requested.append(provider)
            return {
                "api_key": f"{provider}-key",
                "base_url": f"https://{provider}.example.invalid/v1",
                "provider": provider,
                "api_mode": "openai_chat",
            }

        with patch(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            side_effect=_mock_resolve,
        ):
            from gateway.run import _try_resolve_fallback_provider

            result = _try_resolve_fallback_provider()

        assert result is not None
        assert result["provider"] == "openrouter"
        assert result["model"] == "x-ai/grok-4.5"
        assert requested == ["openrouter"]


