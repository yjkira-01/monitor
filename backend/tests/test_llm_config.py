import pytest

from app.llm import config, update_config


def test_runtime_llm_config_masks_and_clears_key(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    saved = update_config(base_url="https://example.com/v1", model="demo", api_key="sk-test-1234567890")
    assert saved["configured"] is True
    assert saved["api_key_masked"] == "sk-t••••••••7890"
    assert "1234567890" not in str(saved)

    cleared = update_config(base_url="https://example.com/v1", model="demo", clear_api_key=True)
    assert cleared["configured"] is False
    assert config()["api_key_masked"] is None

    with pytest.raises(ValueError):
        update_config(base_url="ftp://example.com/v1", model="demo")
