import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock
import pytest


@pytest.mark.asyncio
async def test_resolve_log_path():
    """Test the _resolve_log_path helper function."""
    from proxy.server import _resolve_log_path
    
    proxy_path = _resolve_log_path("proxy")
    assert "proxy.log" in str(proxy_path)
    
    llama_path = _resolve_log_path("llama")
    assert "llama-server.log" in str(llama_path)


@pytest.mark.asyncio
async def test_resolve_log_path_default():
    """Test that default source is proxy."""
    from proxy.server import _resolve_log_path
    
    # Invalid source should default to proxy
    default_path = _resolve_log_path("invalid")
    assert "proxy.log" in str(default_path)
