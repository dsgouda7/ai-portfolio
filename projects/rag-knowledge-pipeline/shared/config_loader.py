"""Configuration loader for the RAG pipeline."""

import os
import yaml
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config.yaml file

    Returns:
        Configuration dictionary

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is malformed
    """
    config_file = Path(config_path)

    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    # Resolve environment variables in config
    config = _resolve_env_vars(config)

    return config


def _resolve_env_vars(config: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively resolve environment variable references in config."""
    if isinstance(config, dict):
        return {k: _resolve_env_vars(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [_resolve_env_vars(item) for item in config]
    elif isinstance(config, str) and config.startswith("${") and config.endswith("}"):
        env_var = config[2:-1]
        return os.getenv(env_var, config)
    else:
        return config


def get_mode(config: Dict[str, Any]) -> str:
    """Extract execution mode from config (local or remote)."""
    return config.get("mode", "local")


def get_local_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract local mode configuration."""
    return config.get("local", {})


def get_remote_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract remote mode configuration."""
    return config.get("remote", {})


def get_llm_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract LLM configuration."""
    return config.get("llm", {})


def get_retrieval_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract retrieval configuration."""
    return config.get("retrieval", {})
