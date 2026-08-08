"""Tests for Bug #12905 fixes in agent/anthropic_adapter.py — macOS Keychain support."""

import json
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

import pytest

from agent.anthropic_adapter import (
    _read_claude_code_credentials_from_file,
    _read_claude_code_credentials_from_keychain,
    _refresh_oauth_token,
    _write_claude_code_credentials,
    read_claude_code_credentials,
)


# This module exercises the reader itself with explicit platform and subprocess
# mocks, so it opts out of the suite-wide guard without touching a real Keychain.
pytestmark = pytest.mark.allow_macos_keychain


@pytest.fixture(autouse=True)
def _isolate_claude_config_dir(monkeypatch):
    """Legacy cases exercise the default profile unless they opt in explicitly."""
    monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)


class TestReadClaudeCodeCredentialsFromKeychain:
    """Bug 4: macOS Keychain support for Claude Code >=2.1.114."""

    def test_returns_none_when_security_command_not_found(self):
        """OSError from missing security binary must be handled gracefully."""
        with patch("agent.anthropic_adapter.platform.system", return_value="Darwin"), \
             patch("agent.anthropic_adapter.subprocess.run",
                   side_effect=OSError("security not found")):
            assert _read_claude_code_credentials_from_keychain() is None

    def test_returns_none_on_nonzero_exit_code(self):
        """security returns non-zero when the Keychain entry doesn't exist."""
        with patch("agent.anthropic_adapter.platform.system", return_value="Darwin"), \
             patch("agent.anthropic_adapter.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="")
            assert _read_claude_code_credentials_from_keychain() is None

    def test_explicit_config_dir_uses_profile_keychain_service(self, monkeypatch):
        """Claude Code hashes the raw explicit config-dir value into the service."""
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", "/Users/octo/.claude-personal")
        payload = json.dumps({
            "claudeAiOauth": {
                "accessToken": "profile-access-token",
                "refreshToken": "profile-refresh-token",
                "expiresAt": 9_999_999_999_999,
            }
        })

        with patch("agent.anthropic_adapter.platform.system", return_value="Darwin"), \
             patch("agent.anthropic_adapter.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout=payload, stderr="")
            creds = _read_claude_code_credentials_from_keychain()

        assert creds is not None
        assert creds["accessToken"] == "profile-access-token"
        command = mock_run.call_args.args[0]
        assert command[command.index("-s") + 1] == "Claude Code-credentials-d02e883c"

    def test_current_user_account_is_preferred_over_duplicate_service_entry(self, monkeypatch):
        """Scope the first lookup so a duplicate empty-account row cannot win."""
        monkeypatch.setenv("USER", "current-user")
        account_payload = json.dumps({
            "claudeAiOauth": {
                "accessToken": "account-scoped-token",
                "expiresAt": 9_999_999_999_999,
            }
        })
        stale_service_payload = json.dumps({
            "claudeAiOauth": {
                "accessToken": "stale-service-token",
                "expiresAt": 1,
            }
        })

        def fake_security(command, **kwargs):
            if "-a" in command:
                return MagicMock(returncode=0, stdout=account_payload, stderr="")
            return MagicMock(returncode=0, stdout=stale_service_payload, stderr="")

        with patch("agent.anthropic_adapter.platform.system", return_value="Darwin"), \
             patch("pwd.getpwuid", return_value=SimpleNamespace(pw_name="current-user")), \
             patch("agent.anthropic_adapter.subprocess.run", side_effect=fake_security) as mock_run:
            creds = _read_claude_code_credentials_from_keychain()

        assert creds is not None
        assert creds["accessToken"] == "account-scoped-token"
        assert mock_run.call_count == 1
        command = mock_run.call_args.args[0]
        assert command[command.index("-a") + 1] == "current-user"

    @pytest.mark.parametrize("environment_user", [None, "spoofed-user"])
    def test_effective_uid_account_ignores_missing_or_mismatched_user_env(
        self, monkeypatch, environment_user
    ):
        """Keychain account scoping must come from the effective OS identity."""
        if environment_user is None:
            monkeypatch.delenv("USER", raising=False)
        else:
            monkeypatch.setenv("USER", environment_user)

        payload = json.dumps({
            "claudeAiOauth": {
                "accessToken": "effective-user-token",
                "expiresAt": 9_999_999_999_999,
            }
        })

        with patch("agent.anthropic_adapter.platform.system", return_value="Darwin"), \
             patch("agent.anthropic_adapter.os.geteuid", return_value=501), \
             patch("pwd.getpwuid", return_value=SimpleNamespace(pw_name="uid-user")), \
             patch("agent.anthropic_adapter.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout=payload, stderr="")
            creds = _read_claude_code_credentials_from_keychain()

        assert creds is not None
        assert creds["accessToken"] == "effective-user-token"
        command = mock_run.call_args.args[0]
        assert command[command.index("-a") + 1] == "uid-user"

    def test_unavailable_os_account_ignores_user_and_logname_environment(self, monkeypatch):
        """Untrusted environment identities must never become Keychain selectors."""
        monkeypatch.setenv("USER", "malicious-user")
        monkeypatch.setenv("LOGNAME", "mismatched-user")
        service_payload = json.dumps({
            "claudeAiOauth": {
                "accessToken": "same-service-token",
                "expiresAt": 9_999_999_999_999,
            }
        })

        with patch("agent.anthropic_adapter.platform.system", return_value="Darwin"), \
             patch("pwd.getpwuid", side_effect=KeyError("unknown effective uid")), \
             patch("agent.anthropic_adapter.os.getlogin", side_effect=OSError("no login")), \
             patch("agent.anthropic_adapter.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0, stdout=service_payload, stderr=""
            )
            creds = _read_claude_code_credentials_from_keychain()

        assert creds is not None
        assert creds["accessToken"] == "same-service-token"
        mock_run.assert_called_once()
        command = mock_run.call_args.args[0]
        assert command[command.index("-s") + 1] == "Claude Code-credentials"
        assert "-a" not in command

    def test_service_only_lookup_is_fallback_when_account_lookup_fails(self, monkeypatch):
        """Keep legacy entries usable when no current-user row can be read."""
        monkeypatch.setenv("USER", "current-user")
        service_payload = json.dumps({
            "claudeAiOauth": {
                "accessToken": "service-fallback-token",
                "expiresAt": 9_999_999_999_999,
            }
        })

        with patch("agent.anthropic_adapter.platform.system", return_value="Darwin"), \
             patch("pwd.getpwuid", return_value=SimpleNamespace(pw_name="current-user")), \
             patch("agent.anthropic_adapter.subprocess.run") as mock_run:
            mock_run.side_effect = [
                MagicMock(returncode=44, stdout="", stderr="not found"),
                MagicMock(returncode=0, stdout=service_payload, stderr=""),
            ]
            creds = _read_claude_code_credentials_from_keychain()

        assert creds is not None
        assert creds["accessToken"] == "service-fallback-token"
        assert mock_run.call_count == 2
        scoped_command = mock_run.call_args_list[0].args[0]
        fallback_command = mock_run.call_args_list[1].args[0]
        assert scoped_command[scoped_command.index("-a") + 1] == "current-user"
        assert "-a" not in fallback_command

    def test_explicit_config_dir_does_not_fall_back_to_default_keychain(self, monkeypatch):
        """A missing profile entry must not silently select another profile."""
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", "/Users/octo/.claude-personal")

        def fake_security(command, **kwargs):
            service = command[command.index("-s") + 1]
            if service == "Claude Code-credentials":
                return MagicMock(
                    returncode=0,
                    stdout=json.dumps({
                        "claudeAiOauth": {
                            "accessToken": "wrong-default-token",
                            "expiresAt": 9_999_999_999_999,
                        }
                    }),
                    stderr="",
                )
            return MagicMock(returncode=1, stdout="", stderr="")

        with patch("agent.anthropic_adapter.platform.system", return_value="Darwin"), \
             patch("agent.anthropic_adapter.subprocess.run", side_effect=fake_security) as mock_run:
            assert _read_claude_code_credentials_from_keychain() is None

        assert mock_run.call_count == 2
        services = [
            call.args[0][call.args[0].index("-s") + 1]
            for call in mock_run.call_args_list
        ]
        assert services == [
            "Claude Code-credentials-d02e883c",
            "Claude Code-credentials-d02e883c",
        ]


class TestClaudeConfigDirCredentialsFile:
    def test_reads_credentials_from_explicit_config_dir(self, tmp_path, monkeypatch):
        config_dir = tmp_path / "claude-personal"
        config_dir.mkdir()
        (config_dir / ".credentials.json").write_text(json.dumps({
            "claudeAiOauth": {
                "accessToken": "profile-file-token",
                "refreshToken": "profile-file-refresh",
                "expiresAt": 9_999_999_999_999,
            }
        }))

        fake_home = tmp_path / "home"
        default_dir = fake_home / ".claude"
        default_dir.mkdir(parents=True)
        (default_dir / ".credentials.json").write_text(json.dumps({
            "claudeAiOauth": {
                "accessToken": "wrong-default-file-token",
                "expiresAt": 9_999_999_999_999,
            }
        }))
        monkeypatch.setattr("agent.anthropic_adapter.Path.home", lambda: fake_home)
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(config_dir))

        creds = _read_claude_code_credentials_from_file()

        assert creds is not None
        assert creds["accessToken"] == "profile-file-token"

    def test_refresh_write_uses_explicit_config_dir(self, tmp_path, monkeypatch):
        config_dir = tmp_path / "claude-personal"
        fake_home = tmp_path / "home"
        monkeypatch.setattr("agent.anthropic_adapter.Path.home", lambda: fake_home)
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(config_dir))

        _write_claude_code_credentials(
            "new-profile-access-token",
            "new-profile-refresh-token",
            9_999_999_999_999,
        )

        assert (config_dir / ".credentials.json").exists()
        assert not (fake_home / ".claude" / ".credentials.json").exists()


class TestReadClaudeCodeCredentialsPriority:
    """Bug 4: Keychain must be checked before the JSON file."""

    def test_keychain_takes_priority_over_json_file(self, tmp_path, monkeypatch):
        """When both Keychain and JSON file have credentials, Keychain wins."""
        # Set up JSON file with "older" token
        json_cred_file = tmp_path / ".claude" / ".credentials.json"
        json_cred_file.parent.mkdir(parents=True)
        json_cred_file.write_text(json.dumps({
            "claudeAiOauth": {
                "accessToken": "json-token",
                "refreshToken": "json-refresh",
                "expiresAt": 9999999999999,
            }
        }))
        monkeypatch.setattr("agent.anthropic_adapter.Path.home", lambda: tmp_path)

        # Mock Keychain to return a "newer" token
        with patch("agent.anthropic_adapter.platform.system", return_value="Darwin"), \
             patch("agent.anthropic_adapter.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout=json.dumps({
                    "claudeAiOauth": {
                        "accessToken": "keychain-token",
                        "refreshToken": "keychain-refresh",
                        "expiresAt": 9999999999999,
                    }
                }),
                stderr="",
            )
            creds = read_claude_code_credentials()

        # Keychain token should be returned, not JSON file token
        assert creds is not None
        assert creds["accessToken"] == "keychain-token"
        assert creds["source"] == "macos_keychain"

    def test_falls_back_to_json_when_keychain_returns_none(self, tmp_path, monkeypatch):
        """When Keychain has no entry, JSON file is used as fallback."""
        json_cred_file = tmp_path / ".claude" / ".credentials.json"
        json_cred_file.parent.mkdir(parents=True)
        json_cred_file.write_text(json.dumps({
            "claudeAiOauth": {
                "accessToken": "json-fallback-token",
                "refreshToken": "json-refresh",
                "expiresAt": 9999999999999,
            }
        }))
        monkeypatch.setattr("agent.anthropic_adapter.Path.home", lambda: tmp_path)

        with patch("agent.anthropic_adapter.platform.system", return_value="Darwin"), \
             patch("agent.anthropic_adapter.subprocess.run") as mock_run:
            # Simulate Keychain entry not found
            mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="")
            creds = read_claude_code_credentials()

        assert creds is not None
        assert creds["accessToken"] == "json-fallback-token"
        assert creds["source"] == "claude_code_credentials_file"

    def test_returns_none_when_neither_keychain_nor_json_has_creds(self, tmp_path, monkeypatch):
        """No credentials anywhere — must return None cleanly."""
        monkeypatch.setattr("agent.anthropic_adapter.Path.home", lambda: tmp_path)

        with patch("agent.anthropic_adapter.platform.system", return_value="Darwin"), \
             patch("agent.anthropic_adapter.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="")
            creds = read_claude_code_credentials()

        assert creds is None


class TestReadClaudeCodeCredentialsDesync:
    """Reconciliation when Keychain and JSON file disagree.

    Observed in the wild on Claude Code 2.1.x: a refresh updates one source
    (commonly the JSON file) but leaves the other holding an expired token.
    The reader must not blindly return whichever source it consulted first;
    it must prefer the non-expired credential.
    """

    # Far-future ms-epoch — comfortably valid under is_claude_code_token_valid.
    _FRESH = 9_999_999_999_999
    # Past ms-epoch — comfortably expired (with the 60s buffer).
    _EXPIRED = 1

    def _setup(self, tmp_path, monkeypatch, *, file_expires_at, file_token="json-token"):
        json_cred_file = tmp_path / ".claude" / ".credentials.json"
        json_cred_file.parent.mkdir(parents=True)
        json_cred_file.write_text(json.dumps({
            "claudeAiOauth": {
                "accessToken": file_token,
                "refreshToken": "json-refresh",
                "expiresAt": file_expires_at,
            }
        }))
        monkeypatch.setattr("agent.anthropic_adapter.Path.home", lambda: tmp_path)

    def _keychain_payload(self, *, access_token, expires_at, refresh_token="kc-refresh"):
        return MagicMock(
            returncode=0,
            stdout=json.dumps({
                "claudeAiOauth": {
                    "accessToken": access_token,
                    "refreshToken": refresh_token,
                    "expiresAt": expires_at,
                }
            }),
            stderr="",
        )

    def test_keychain_expired_file_fresh_returns_file(self, tmp_path, monkeypatch):
        """Regression: when the Keychain holds an expired token but the JSON
        file has a valid one, callers must receive the valid file token rather
        than None. (Pre-fix behavior returned the expired Keychain token, and
        downstream validity checks then yielded None — surfacing the misleading
        ``No Anthropic credentials found`` error.)
        """
        self._setup(tmp_path, monkeypatch, file_expires_at=self._FRESH, file_token="fresh-file-token")
        with patch("agent.anthropic_adapter.platform.system", return_value="Darwin"), \
             patch("agent.anthropic_adapter.subprocess.run") as mock_run:
            mock_run.return_value = self._keychain_payload(
                access_token="stale-keychain-token", expires_at=self._EXPIRED,
            )
            creds = read_claude_code_credentials()

        assert creds is not None
        assert creds["accessToken"] == "fresh-file-token"
        assert creds["source"] == "claude_code_credentials_file"



    def test_both_expired_prefers_later_expiry(self, tmp_path, monkeypatch):
        """When both are expired, return the one with the later ``expiresAt``;
        its ``refresh_token`` is the most recently issued and most likely to
        succeed at the OAuth refresh endpoint.
        """
        self._setup(tmp_path, monkeypatch, file_expires_at=self._EXPIRED + 5, file_token="newer-expired-file")
        with patch("agent.anthropic_adapter.platform.system", return_value="Darwin"), \
             patch("agent.anthropic_adapter.subprocess.run") as mock_run:
            mock_run.return_value = self._keychain_payload(
                access_token="older-expired-keychain", expires_at=self._EXPIRED,
            )
            creds = read_claude_code_credentials()

        assert creds is not None
        assert creds["accessToken"] == "newer-expired-file"


class TestRefreshOAuthTokenAdoptsFreshCredential:
    """``_refresh_oauth_token`` should adopt a credential Claude Code has
    already refreshed rather than POSTing a (possibly already-rotated)
    single-use refresh token and racing Claude Code into ``invalid_grant``.
    """

    _FRESH = 9_999_999_999_999

    def test_adopts_already_refreshed_token_without_posting(self, monkeypatch):
        """When a live source already holds a valid token, return it and skip
        the network refresh entirely.
        """
        fresh = {
            "accessToken": "already-refreshed-token",
            "refreshToken": "live-refresh",
            "expiresAt": self._FRESH,
        }
        monkeypatch.setattr(
            "agent.anthropic_adapter.read_claude_code_credentials",
            lambda: fresh,
        )

        def _should_not_be_called(*args, **kwargs):  # pragma: no cover - guard
            raise AssertionError("refresh_anthropic_oauth_pure must not be called")

        monkeypatch.setattr(
            "agent.anthropic_adapter.refresh_anthropic_oauth_pure",
            _should_not_be_called,
        )

        # Stale creds passed in by the caller — should be ignored in favor
        # of the live, already-refreshed token.
        result = _refresh_oauth_token({"refreshToken": "stale", "expiresAt": 1})
        assert result == "already-refreshed-token"

    def test_falls_back_to_network_refresh_when_no_fresh_credential(self, monkeypatch):
        """When no live source has a valid token, fall back to refreshing
        ourselves using the freshest available refresh token.
        """
        # Live read returns an expired credential carrying a refresh token.
        monkeypatch.setattr(
            "agent.anthropic_adapter.read_claude_code_credentials",
            lambda: {"accessToken": "expired", "refreshToken": "live-refresh", "expiresAt": 1},
        )
        captured = {}

        def _fake_refresh(refresh_token, **kwargs):
            captured["refresh_token"] = refresh_token
            return {
                "access_token": "newly-minted",
                "refresh_token": "rotated",
                "expires_at_ms": self._FRESH,
            }

        monkeypatch.setattr(
            "agent.anthropic_adapter.refresh_anthropic_oauth_pure", _fake_refresh
        )
        monkeypatch.setattr(
            "agent.anthropic_adapter._write_claude_code_credentials",
            lambda *a, **k: None,
        )

        result = _refresh_oauth_token({"refreshToken": "caller-refresh", "expiresAt": 1})
        assert result == "newly-minted"
        # Prefers the live source's refresh token over the caller's stale copy.
        assert captured["refresh_token"] == "live-refresh"

