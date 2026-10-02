"""Exercise real Floret predictions without downloading a language model."""

import json

import floret
import pytest

from impresso_pipelines.langident import LangIdentPipeline


@pytest.fixture(scope="module")
def local_pipeline(tmp_path_factory):
    training = tmp_path_factory.mktemp("floret") / "training.txt"
    training.write_text(
        "__label__en the church and the school in the town\n"
        "__label__en funeral services were held on Tuesday\n"
        "__label__de die Kirche und die Schule in der Stadt\n"
        "__label__de die Familie lebt in der Stadt\n",
        encoding="utf-8",
    )
    model = floret.train_supervised(
        input=str(training), dim=8, epoch=5, minCount=1, thread=1, verbose=0
    )
    pipeline = LangIdentPipeline.__new__(LangIdentPipeline)
    pipeline.model = model
    pipeline.model_name = "local-test-model"
    return pipeline


def test_real_floret_predictions_are_json_serializable(local_pipeline):
    for diagnostics in (False, True):
        text = "Funeral services\nwere held at the church."
        labels, scores = local_pipeline.model.predict(
            [text.replace("\n", " ")], k=300 if diagnostics else 1
        )
        result = local_pipeline(text, diagnostics=diagnostics, model_id=True)

        assert result["language"] == labels[0][0].replace("__label__", "")
        assert result["score"] == round(float(scores[0][0]), 2)
        assert result["model_id"] == "local-test-model"
        assert json.loads(json.dumps(result)) == result
        if diagnostics:
            assert result["diagnostics"]["languages"] == [
                {
                    "language": label.replace("__label__", ""),
                    "score": round(float(score), 2),
                }
                for label, score in zip(labels[0], scores[0])
            ]
