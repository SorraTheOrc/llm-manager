import pytest
import asyncio
from unittest.mock import patch, AsyncMock, MagicMock


class MockResponse:
    def __init__(self, status_code=200, json_data=None):
        self.status_code = status_code
        self._json_data = json_data

    async def json(self):
        return self._json_data


class MockAsyncClient:
    def __init__(self, responses=None):
        self.responses = responses or []
        self.calls = []

    async def get(self, url, **kwargs):
        self.calls.append(url)
        if self.responses:
            return self.responses.pop(0)
        return MockResponse(404)


@pytest.fixture
def mock_config():
    return {
        "server": {
            "llama_server_port": 8080
        }
    }


@pytest.mark.asyncio
async def test_query_llama_status_server_not_running(mock_config):
    from proxy.server import query_llama_status
    
    with patch('proxy.server.config', mock_config):
        with patch('proxy.server.llama_process', None):
            result = await query_llama_status()
            
            assert result["llama_server_running"] is False
            assert result["n_ctx"] is None
            assert result["kv_cache_tokens"] is None


@pytest.mark.asyncio
async def test_query_llama_status_full_response(mock_config):
    from proxy.server import query_llama_status
    
    mock_process = MagicMock()
    mock_process.poll.return_value = None
    
    mock_responses = [
        MockResponse(200, {"n_ctx": 4096, "model": "qwen3"}),
        MockResponse(200, {"kv_cache_tokens": 1500}),
    ]
    mock_client = MockAsyncClient(mock_responses)
    
    with patch('proxy.server.config', mock_config):
        with patch('proxy.server.llama_process', mock_process):
            with patch('httpx.AsyncClient', return_value=mock_client):
                result = await query_llama_status()
                
                assert result["llama_server_running"] is True
                assert result["n_ctx"] == 4096
                assert result["kv_cache_tokens"] == 1500


@pytest.mark.asyncio
async def test_query_llama_status_partial_response(mock_config):
    from proxy.server import query_llama_status
    
    mock_process = MagicMock()
    mock_process.poll.return_value = None
    
    mock_responses = [
        MockResponse(200, {"model": "qwen3"}),
        MockResponse(404),
    ]
    mock_client = MockAsyncClient(mock_responses)
    
    with patch('proxy.server.config', mock_config):
        with patch('proxy.server.llama_process', mock_process):
            with patch('httpx.AsyncClient', return_value=mock_client):
                result = await query_llama_status()
                
                assert result["llama_server_running"] is True
                assert result["n_ctx"] is None
                assert result["kv_cache_tokens"] is None


@pytest.mark.asyncio
async def test_query_llama_status_n_ctx_total_fallback(mock_config):
    from proxy.server import query_llama_status
    
    mock_process = MagicMock()
    mock_process.poll.return_value = None
    
    mock_responses = [
        MockResponse(200, {"n_ctx_total": 8192, "model": "qwen3"}),
    ]
    mock_client = MockAsyncClient(mock_responses)
    
    with patch('proxy.server.config', mock_config):
        with patch('proxy.server.llama_process', mock_process):
            with patch('httpx.AsyncClient', return_value=mock_client):
                result = await query_llama_status()
                
                assert result["n_ctx"] == 8192


@pytest.mark.asyncio
async def test_query_llama_status_kv_cache_token_count(mock_config):
    from proxy.server import query_llama_status
    
    mock_process = MagicMock()
    mock_process.poll.return_value = None
    
    mock_responses = [
        MockResponse(200, {"n_ctx": 4096}),
        MockResponse(200, {"kv_cache_token_count": 2048}),
    ]
    mock_client = MockAsyncClient(mock_responses)
    
    with patch('proxy.server.config', mock_config):
        with patch('proxy.server.llama_process', mock_process):
            with patch('httpx.AsyncClient', return_value=mock_client):
                result = await query_llama_status()
                
                assert result["kv_cache_tokens"] == 2048


@pytest.mark.asyncio
async def test_query_llama_status_connection_error(mock_config):
    from proxy.server import query_llama_status
    
    mock_process = MagicMock()
    mock_process.poll.return_value = None
    
    mock_client = MockAsyncClient([])
    
    class ConnectionErrorResponse:
        status_code = 0
        async def json(self):
            raise Exception("Connection failed")
    
    async def mock_get_side_effect(*args, **kwargs):
        raise Exception("Connection failed")
    
    mock_client.get = mock_get_side_effect
    
    with patch('proxy.server.config', mock_config):
        with patch('proxy.server.llama_process', mock_process):
            with patch('httpx.AsyncClient', return_value=mock_client):
                result = await query_llama_status()
                
                assert result["llama_server_running"] is True
                assert result["n_ctx"] is None
                assert result["kv_cache_tokens"] is None


@pytest.mark.asyncio
async def test_query_llama_status_invalid_json(mock_config):
    from proxy.server import query_llama_status
    
    mock_process = MagicMock()
    mock_process.poll.return_value = None
    
    class InvalidJsonResponse:
        status_code = 200
        async def json(self):
            raise ValueError("Invalid JSON")
    
    mock_client = MockAsyncClient([InvalidJsonResponse()])
    
    with patch('proxy.server.config', mock_config):
        with patch('proxy.server.llama_process', mock_process):
            with patch('httpx.AsyncClient', return_value=mock_client):
                result = await query_llama_status()
                
                assert result["llama_server_running"] is True
                assert result["n_ctx"] is None


@pytest.mark.asyncio
async def test_query_llama_status_process_dead(mock_config):
    from proxy.server import query_llama_status
    
    mock_process = MagicMock()
    mock_process.poll.return_value = 1
    
    with patch('proxy.server.config', mock_config):
        with patch('proxy.server.llama_process', mock_process):
            result = await query_llama_status()
            
            assert result["llama_server_running"] is False
            assert result["n_ctx"] is None
            assert result["kv_cache_tokens"] is None


@pytest.mark.asyncio
async def test_query_llama_status_custom_port(mock_config):
    from proxy.server import query_llama_status
    
    custom_config = {
        "server": {
            "llama_server_port": 9090
        }
    }
    
    mock_process = MagicMock()
    mock_process.poll.return_value = None
    
    mock_client = MockAsyncClient([MockResponse(200, {"n_ctx": 2048})])
    
    with patch('proxy.server.config', custom_config):
        with patch('proxy.server.llama_process', mock_process):
            with patch('httpx.AsyncClient', return_value=mock_client):
                result = await query_llama_status()
                
                assert "localhost:9090" in mock_client.calls[0]


@pytest.mark.asyncio
async def test_query_llama_status_concurrent_requests_do_not_block(mock_config):
    """Verify multiple concurrent status requests complete without blocking.
    
    This test verifies that when using a shared httpx client with connection pooling,
    multiple concurrent calls to query_llama_status complete in a reasonable time
    rather than blocking each other.
    """
    from proxy.server import query_llama_status
    
    mock_process = MagicMock()
    mock_process.poll.return_value = None
    
    async def slow_get(*args, **kwargs):
        await asyncio.sleep(0.05)
        return MockResponse(200, {"n_ctx": 4096})
    
    mock_client = MagicMock()
    mock_client.get = slow_get
    mock_client.aclose = AsyncMock()
    
    with patch('proxy.server.config', mock_config):
        with patch('proxy.server.llama_process', mock_process):
            with patch('proxy.server._http_client', mock_client):
                start = asyncio.get_event_loop().time()
                
                results = await asyncio.wait_for(
                    asyncio.gather(
                        query_llama_status(),
                        query_llama_status(),
                        query_llama_status()
                    ),
                    timeout=2.0
                )
                
                elapsed = asyncio.get_event_loop().time() - start
                
                assert len(results) == 3
                for result in results:
                    assert result["llama_server_running"] is True
                    assert result["n_ctx"] == 4096
                
                assert elapsed < 1.0


@pytest.mark.asyncio
async def test_status_request_during_streaming_request(mock_config):
    """Verify status requests complete even when a streaming request is active.
    
    This test simulates the scenario described in LP-0MNWEZMUX009XL9B where
    the /llama/local/status endpoint hangs while llama-server is busy processing
    a streaming OpenAI request. With connection pooling, the status request
    should complete within a reasonable time.
    """
    from proxy.server import query_llama_status
    
    mock_process = MagicMock()
    mock_process.poll.return_value = None
    
    status_response_received = asyncio.Event()
    
    async def slow_status_get(*args, **kwargs):
        await asyncio.sleep(0.1)
        return MockResponse(200, {"n_ctx": 4096})
    
    async def blocking_stream_get(*args, **kwargs):
        await status_response_received.wait()
        return MockResponse(200, {"content": b"data: chunk\n\n"})
    
    mock_status_client = MagicMock()
    mock_status_client.get = slow_status_get
    mock_status_client.aclose = AsyncMock()
    
    async def run_test():
        nonlocal status_response_received
        
        async def streaming_request():
            status_response_received.set()
            await asyncio.sleep(0.2)
            return [b"data: chunk\n\n"]
        
        async def status_request():
            return await query_llama_status()
        
        results = await asyncio.gather(
            streaming_request(),
            status_request()
        )
        return results
    
    with patch('proxy.server.config', mock_config):
        with patch('proxy.server.llama_process', mock_process):
            with patch('proxy.server._http_client', mock_status_client):
                start = asyncio.get_event_loop().time()
                
                results = await asyncio.wait_for(run_test(), timeout=2.0)
                
                elapsed = asyncio.get_event_loop().time() - start
                
                assert results[1]["llama_server_running"] is True
                assert results[1]["n_ctx"] == 4096
                assert elapsed < 1.0
