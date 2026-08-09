"""Tests for hermes_cli/fallback_config.py — fallback entry API-key resolution."""

from agent.secret_scope import reset_secret_scope, set_secret_scope
from hermes_cli.fallback_config import (
    get_fallback_chain,
    get_runtime_fallback_chain,
    resolve_entry_api_key,
)


class TestResolveEntryApiKey:
    def test_inline_api_key_wins(self, monkeypatch):
        monkeypatch.setenv("FB_KEY", "env-key")
        entry = {"provider": "custom", "api_key": "inline-key", "key_env": "FB_KEY"}
        assert resolve_entry_api_key(entry) == "inline-key"


    def test_no_key_fields_returns_none(self):
        assert resolve_entry_api_key({"provider": "openrouter", "model": "glm"}) is None


    def test_whitespace_inline_key_falls_through_to_env(self, monkeypatch):
        monkeypatch.setenv("FB_KEY", "env-key")
        entry = {"api_key": "   ", "key_env": "FB_KEY"}
        assert resolve_entry_api_key(entry) == "env-key"

    def test_key_env_resolves_from_active_secret_scope_not_raw_env(self, monkeypatch):
        # Multiplexed gateway: os.environ holds another profile's key, but the
        # active per-turn secret scope holds this profile's key. The scoped
        # value must win — a raw os.getenv() would leak the other profile's
        # credential (issue #74311).
        monkeypatch.setenv("FB_KEY", "fake-other-profile-key")
        token = set_secret_scope({"FB_KEY": "fake-active-profile-key"})
        try:
            assert resolve_entry_api_key({"key_env": "FB_KEY"}) == "fake-active-profile-key"
        finally:
            reset_secret_scope(token)

    def test_key_env_falls_back_to_env_when_no_active_scope(self, monkeypatch):
        # Non-multiplexed / single-profile behavior must be unchanged: with no
        # secret scope installed, resolution still reads os.environ.
        monkeypatch.setenv("FB_KEY", "env-key")
        assert resolve_entry_api_key({"key_env": "FB_KEY"}) == "env-key"


class TestRuntimeFallbackChain:
    CONFIG = {
        "fallback_providers": [
            {"provider": "openai-codex", "model": "gpt-5.6-sol"},
            {"provider": "xai", "model": "grok-4.5"},
        ]
    }

    def test_declared_chain_keeps_codex_for_display_and_introspection(self):
        assert [entry["provider"] for entry in get_fallback_chain(self.CONFIG)] == [
            "openai-codex",
            "xai",
        ]

    def test_parent_runtime_chain_skips_codex_and_preserves_ordinary_entries(self):
        assert [
            entry["provider"] for entry in get_runtime_fallback_chain(self.CONFIG)
        ] == ["xai"]

    def test_delegated_child_runtime_chain_preserves_codex_sol(self):
        from agent.delegation_context import delegated_child_context

        with delegated_child_context():
            chain = get_runtime_fallback_chain(self.CONFIG)

        assert chain[0] == {
            "provider": "openai-codex",
            "model": "gpt-5.6-sol",
        }
        assert [entry["provider"] for entry in chain] == ["openai-codex", "xai"]
