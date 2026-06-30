import importlib.util
from pathlib import Path


def load_ldatopics_config():
    config_path = (
        Path(__file__).resolve().parents[2]
        / "impresso_pipelines"
        / "ldatopics"
        / "config.py"
    )
    spec = importlib.util.spec_from_file_location("ldatopics_config", config_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_english_language_config_is_registered():
    config = load_ldatopics_config()

    assert config.SUPPORTED_LANGUAGES["en"] == "en_core_web_md"
    assert config.MODEL_URLS["en_core_web_md"].endswith(
        "/en_core_web_md-3.6.0/en_core_web_md-3.6.0.tar.gz"
    )
    assert (
        config.TOPIC_MODEL_DESCRIPTIONS["en"]
        == "https://huggingface.co/impresso-project/mallet-topic-inferencer/resolve/main/models/tm/tm-en-all-v3.0.topic_model_topic_description.jsonl.bz2"
    )
    assert config.TOPIC_MODEL_DESCRIPTIONS_HF["en"] == [
        "impresso-project/mallet-topic-inferencer",
        "models/tm/tm-en-all-v3.0.topic_model_topic_description.jsonl.bz2",
    ]
