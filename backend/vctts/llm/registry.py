"""Provider registry: builds providers on demand from stored API keys."""

from __future__ import annotations

from typing import Callable

from ..secrets import SecretStore
from .base import ChatProvider, ProviderError

ProviderFactory = Callable[[str], ChatProvider]


def _factories() -> dict[str, tuple[type[ChatProvider], ProviderFactory]]:
    # Imported lazily so a broken optional SDK only disables its own provider.
    from .anthropic_provider import AnthropicProvider
    from .gemini_provider import GeminiProvider
    from .openai_compat import OpenAIProvider, OpenRouterProvider

    return {
        cls.id: (cls, cls)  # type: ignore[misc]
        for cls in (OpenAIProvider, AnthropicProvider, GeminiProvider, OpenRouterProvider)
    }


class ProviderRegistry:
    def __init__(self, secrets: SecretStore):
        self.secrets = secrets
        self._factories = _factories()
        self._cache: dict[str, tuple[str, ChatProvider]] = {}
        self._overrides: dict[str, ChatProvider] = {}

    def ids(self) -> list[str]:
        return list(self._factories)

    def describe(self) -> list[dict]:
        out = []
        for pid, (cls, _) in self._factories.items():
            out.append(
                {
                    "id": pid,
                    "name": cls.display_name,
                    "default_model": cls.default_model,
                    "suggested_models": list(cls.suggested_models),
                    "configured": pid in self._overrides or self.secrets.has(pid),
                    "key_source": self.secrets.source(pid),
                }
            )
        return out

    def register_override(self, pid: str, provider: ChatProvider) -> None:
        """Inject a provider instance (used by tests and future local backends)."""
        self._overrides[pid] = provider

    def get(self, pid: str) -> ChatProvider:
        if pid in self._overrides:
            return self._overrides[pid]
        if pid not in self._factories:
            raise ProviderError(f"Unknown provider '{pid}'")
        key = self.secrets.get(pid)
        if not key:
            name = self._factories[pid][0].display_name
            raise ProviderError(f"Add your {name} API key in Settings first.")
        cached = self._cache.get(pid)
        if cached and cached[0] == key:
            return cached[1]
        provider = self._factories[pid][1](key)
        self._cache[pid] = (key, provider)
        return provider

    def invalidate(self, pid: str) -> None:
        self._cache.pop(pid, None)
