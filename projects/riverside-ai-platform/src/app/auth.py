from __future__ import annotations

import asyncio
import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

import jwt
from jwt import PyJWKClient
from jwt.exceptions import PyJWTError


class BackendAuthenticationError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class BackendIdentityConfig:
    tenant_id: str
    audience: str
    apim_principal_id: str
    authority_host: str = "https://login.microsoftonline.com"

    def __post_init__(self) -> None:
        if not self.tenant_id or not self.audience or not self.apim_principal_id:
            raise ValueError("backend tenant, audience, and APIM principal are required")
        parsed = urlparse(self.authority_host)
        if parsed.scheme != "https" or not parsed.netloc or parsed.path not in {"", "/"}:
            raise ValueError("backend authority host must be an HTTPS origin")

    @property
    def jwks_uri(self) -> str:
        return (
            f"{self.authority_host.rstrip('/')}/{self.tenant_id}/"
            "discovery/v2.0/keys"
        )

    @property
    def allowed_issuers(self) -> frozenset[str]:
        return frozenset(
            {
                f"https://sts.windows.net/{self.tenant_id}/",
                f"{self.authority_host.rstrip('/')}/{self.tenant_id}/v2.0",
            }
        )

    @classmethod
    def from_environment(
        cls,
        environ: Mapping[str, str] | None = None,
    ) -> BackendIdentityConfig:
        values = os.environ if environ is None else environ

        def required(name: str) -> str:
            value = values.get(name)
            if not value:
                raise ValueError(f"missing required environment variable: {name}")
            return value

        return cls(
            tenant_id=required("RIVERSIDE_BACKEND_TENANT_ID"),
            audience=required("RIVERSIDE_BACKEND_AUDIENCE"),
            apim_principal_id=required("RIVERSIDE_APIM_PRINCIPAL_ID"),
            authority_host=values.get(
                "AZURE_AUTHORITY_HOST",
                "https://login.microsoftonline.com",
            ),
        )


class BackendIdentityValidator:
    def __init__(self, config: BackendIdentityConfig) -> None:
        self._config = config
        self._jwks = PyJWKClient(config.jwks_uri, cache_jwk_set=True, lifespan=3600)

    async def authenticate(self, authorization: str | None) -> None:
        if not authorization or not authorization.startswith("Bearer "):
            raise BackendAuthenticationError("backend bearer token is unavailable")
        token = authorization.removeprefix("Bearer ").strip()
        if not token:
            raise BackendAuthenticationError("backend bearer token is unavailable")
        try:
            claims = await asyncio.to_thread(self._decode, token)
        except (PyJWTError, ValueError, OSError):
            raise BackendAuthenticationError("backend bearer token is invalid") from None
        issuer = claims.get("iss")
        principal_id = claims.get("oid")
        if issuer not in self._config.allowed_issuers:
            raise BackendAuthenticationError("backend token issuer is invalid")
        if principal_id != self._config.apim_principal_id:
            raise BackendAuthenticationError("backend caller is not the configured APIM identity")

    def _decode(self, token: str) -> dict[str, Any]:
        signing_key = self._jwks.get_signing_key_from_jwt(token)
        claims = jwt.decode(
            token,
            signing_key.key,
            algorithms=["RS256"],
            audience=self._config.audience,
            options={"require": ["exp", "iat", "iss", "aud", "oid"]},
        )
        if not isinstance(claims, dict):
            raise BackendAuthenticationError("backend token claims are invalid")
        return claims
