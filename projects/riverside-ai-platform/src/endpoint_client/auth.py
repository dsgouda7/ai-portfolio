from __future__ import annotations

from typing import Any, Protocol


class AccessToken(Protocol):
    token: str


class AsyncTokenCredential(Protocol):
    async def get_token(self, *scopes: str, **kwargs: Any) -> AccessToken: ...

    async def close(self) -> None: ...


def create_default_credential(managed_identity_client_id: str | None = None) -> AsyncTokenCredential:
    from azure.identity.aio import DefaultAzureCredential

    return DefaultAzureCredential(managed_identity_client_id=managed_identity_client_id)
