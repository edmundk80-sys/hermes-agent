"""Parent/child policy tests for CLI setup-time fallback activation."""

from hermes_cli.cli_agent_setup_mixin import CLIAgentSetupMixin


class _Shell(CLIAgentSetupMixin):
    def __init__(self):
        self.requested_provider = "anthropic"
        self._explicit_api_key = None
        self._explicit_base_url = None
        self._fallback_model = [
            {"provider": "openai-codex", "model": "gpt-5.6-sol"},
            {"provider": "xai", "model": "grok-4.5"},
        ]
        self.model = "claude-fable-5"
        self.provider = "anthropic"
        self.api_mode = "openai_chat"
        self.acp_command = None
        self.acp_args = []
        self._credential_pool = None
        self._provider_source = None
        self.api_key = None
        self.base_url = None
        self.agent = None
        self._active_agent_route_signature = None

    def _normalize_model_for_provider(self, _provider):
        return False


def test_parent_cli_auth_fallback_skips_openai_codex(monkeypatch):
    """CLI parent credential recovery must continue past Codex to XAI."""
    from hermes_cli.auth import AuthError

    requested = []

    def fake_resolve(**kwargs):
        provider = kwargs.get("requested")
        requested.append(provider)
        if provider == "anthropic":
            raise AuthError("No Anthropic credentials stored")
        return {
            "provider": provider,
            "api_key": f"{provider}-key",
            "base_url": f"https://{provider}.example.invalid/v1",
            "api_mode": "openai_chat",
        }

    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        fake_resolve,
    )

    shell = _Shell()
    assert shell._ensure_runtime_credentials() is True
    assert requested == ["anthropic", "xai"]
    assert shell.requested_provider == "xai"
    assert shell.provider == "xai"
    assert shell.model == "grok-4.5"
