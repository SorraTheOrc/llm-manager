import pytest
from proxy.log_parser import detect_severity


def test_detect_severity_basic_prefixes():
    assert detect_severity('ERROR Something bad happened') == 'error'
    assert detect_severity('WARN this might be wrong') == 'warning'
    assert detect_severity('INFO startup complete') == 'info'
    assert detect_severity('DEBUG variable x=1') == 'debug'


def test_detect_severity_brackets():
    assert detect_severity('[ERROR] failed to bind') == 'error'
    assert detect_severity('[WARN] low disk') == 'warning'
    assert detect_severity('[INFO] ready') == 'info'


def test_detect_severity_json_level():
    j = '{"time": "now", "level": "ERROR", "msg": "boom"}'
    assert detect_severity(j) == 'error'
    j2 = '{"severity":"warning","msg":"low"}'
    assert detect_severity(j2) == 'warning'
    j3 = '{"levelname":"INFO","msg":"ok"}'
    assert detect_severity(j3) == 'info'


def test_detect_severity_unknown():
    assert detect_severity('This is a regular log line with no level') == 'unknown'
    assert detect_severity('') == 'unknown'
