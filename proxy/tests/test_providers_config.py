import proxy.server as server


def test_load_config_has_proxy_provider():
    cfg = server.load_config()
    assert isinstance(cfg, dict)
    providers = cfg.get("providers")
    assert isinstance(providers, dict), "providers mapping must exist in config"
    proxy_cfg = providers.get("Proxy")
    assert isinstance(proxy_cfg, dict), "Proxy provider must be configured"
    # Expect full URL including scheme and port
    assert proxy_cfg.get("host") in ("http://100.79.231.101:8000", "https://100.79.231.101:8000", "100.79.231.101:8000"), f"unexpected host: {proxy_cfg.get('host')}"


def test_normalize_provider_name():
    assert server.normalize_provider_name("Local Proxy") == "Proxy"
    assert server.normalize_provider_name("local proxy") == "Proxy"
    assert server.normalize_provider_name("Proxy") == "Proxy"
    assert server.normalize_provider_name("") == ""
    assert server.normalize_provider_name(None) is None
