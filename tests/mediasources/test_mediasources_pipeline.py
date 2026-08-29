from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from impresso_pipelines.mediasources import mediasources_pipeline as mediasources_module
from impresso_pipelines.mediasources.mediasources_pipeline import MediaSourcesPipeline, load_tokenizer, tokenize_with_offsets
from impresso_pipelines.newsagencies import NewsAgenciesPipeline


ID2LABEL = {
    0: "O",
    1: "B-org.ent.pressagency.reuters",
    2: "I-org.ent.pressagency.reuters",
    3: "B-org.ent.radiostation.bbc",
    4: "I-org.ent.radiostation.bbc",
}


class FakeEncoding(dict):
    def __init__(self, token_batches: list[list[str]], label_id_batches: list[list[int]], logits_by_token: list[list[list[float]]] | None = None):
        max_len = max((len(tokens) for tokens in token_batches), default=0)
        super().__init__(
            {
                "input_ids": torch.tensor(
                    [
                        [100 + index for index in range(len(tokens))] + [0] * (max_len - len(tokens))
                        for tokens in token_batches
                    ]
                ),
                "attention_mask": torch.tensor(
                    [[1 for _token in tokens] + [0] * (max_len - len(tokens)) for tokens in token_batches]
                ),
            }
        )
        self._word_ids = [list(range(len(tokens))) + [None] * (max_len - len(tokens)) for tokens in token_batches]
        self.label_ids = label_id_batches
        self.logits_by_token = logits_by_token

    def word_ids(self, batch_index: int = 0) -> list[int | None]:
        return self._word_ids[batch_index]


class FakeTokenizer:
    def __init__(self, label_by_token: dict[str, int], logits_by_token: dict[str, list[float]] | None = None):
        self.label_by_token = label_by_token
        self.logits_by_token = logits_by_token or {}
        self.calls: list[list[str]] = []

    def __call__(self, tokens: list[str] | list[list[str]], **_kwargs: object) -> FakeEncoding:
        token_batches = tokens if tokens and isinstance(tokens[0], list) else [tokens]
        self.calls.extend([list(token_batch) for token_batch in token_batches])
        label_id_batches = [
            [self.label_by_token.get(token, 0) for token in token_batch]
            for token_batch in token_batches
        ]
        has_explicit_logits = all(
            token in self.logits_by_token
            for token_batch in token_batches
            for token in token_batch
        )
        logits = [
            [self.logits_by_token[token] for token in token_batch]
            for token_batch in token_batches
        ] if has_explicit_logits else None
        encoding = FakeEncoding(token_batches, label_id_batches, logits)
        encoding["_label_ids"] = encoding.label_ids
        if logits is not None:
            max_len = max(len(row) for row in logits)
            padded_logits = [
                row + [[0.0 for _ in ID2LABEL] for _pad in range(max_len - len(row))]
                for row in logits
            ]
            encoding["_logits"] = torch.tensor(padded_logits)
        return encoding


class FakeModel:
    def __init__(self, id2label: dict[int, str] | None = None):
        self.config = SimpleNamespace(
            id2label=id2label or ID2LABEL,
            annotation_tokenization="unicode-word-punctuation-v1",
            subtoken_decoding="first_subtoken_viterbi",
            max_sequence_len=512,
            max_annotation_tokens=256,
            stride=32,
        )
        self.device = torch.device("cpu")
        self.batch_sizes: list[int] = []

    def to(self, device):
        self.device = torch.device(device)
        return self

    def eval(self):
        return self

    def __call__(self, **inputs):
        explicit_logits = inputs.pop("_logits", None)
        if explicit_logits is not None:
            self.batch_sizes.append(int(explicit_logits.shape[0]))
            return SimpleNamespace(logits=explicit_logits)
        label_ids = inputs.pop("_label_ids")
        self.batch_sizes.append(len(label_ids))
        max_len = max((len(row) for row in label_ids), default=0)
        logits = torch.full((len(label_ids), max_len, len(ID2LABEL)), -10.0)
        for batch_index, row in enumerate(label_ids):
            for token_index, label_id in enumerate(row):
                logits[batch_index, token_index, label_id] = 10.0
        return SimpleNamespace(logits=logits)


class FakeModernBertModel(FakeModel):
    def __call__(self, input_ids=None, attention_mask=None):
        logits = torch.full((1, input_ids.shape[1], len(ID2LABEL)), -10.0)
        logits[0, :, 1] = 10.0
        return SimpleNamespace(logits=logits)

    def forward(self, input_ids=None, attention_mask=None):
        return self(input_ids=input_ids, attention_mask=attention_mask)


class TokenTypeIdsTokenizer(FakeTokenizer):
    def __call__(self, tokens: list[str], **kwargs: object) -> FakeEncoding:
        encoding = super().__call__(tokens, **kwargs)
        encoding["token_type_ids"] = torch.zeros_like(encoding["input_ids"])
        return encoding


def make_pipeline(
    label_by_token: dict[str, int],
    logits_by_token: dict[str, list[float]] | None = None,
    **kwargs,
) -> MediaSourcesPipeline:
    return MediaSourcesPipeline(FakeModel(), tokenizer=FakeTokenizer(label_by_token, logits_by_token), device=-1, **kwargs)


def test_tokenizer_matches_model_protocol() -> None:
    tokens, starts, stops = tokenize_with_offsets("Selon l'Agence France-Presse.")

    assert tokens == ["Selon", "l", "'", "Agence", "France", "-", "Presse", "."]
    assert [
        "Selon l'Agence France-Presse."[start:stop]
        for start, stop in zip(starts, stops, strict=True)
    ] == tokens


def test_media_sources_pipeline_is_separate_from_legacy_newsagencies_pipeline() -> None:
    assert MediaSourcesPipeline is not NewsAgenciesPipeline


def test_diagnostics_return_decoded_entities_with_exact_offsets() -> None:
    pipe = make_pipeline({"BBC": 3, "World": 4, "Service": 4})

    result = pipe("The BBC World Service reported.", diagnostics=True)

    assert result["token_labels"] == [
        "O",
        "B-org.ent.radiostation.bbc",
        "I-org.ent.radiostation.bbc",
        "I-org.ent.radiostation.bbc",
        "O",
        "O",
    ]
    assert result["entities"] == [
        {
            "label": "org.ent.radiostation.bbc",
            "wkdata_qid": "Q9531",
            "start": 4,
            "stop": 21,
            "surface": "BBC World Service",
            "score": pytest.approx(1.0),
        }
    ]


def test_score_marginalizes_over_bio_state_for_decoded_entity() -> None:
    pipe = make_pipeline(
        {},
        logits_by_token={
            "Reuters": [0.0, 2.0, 4.0, -10.0, -10.0],
        },
    )

    result = pipe("Reuters", diagnostics=True)

    assert result["token_labels"] == ["B-org.ent.pressagency.reuters"]
    assert result["entities"][0]["score"] == pytest.approx(result["token_scores"][0])
    assert result["entities"][0]["score"] > 0.98
    assert result["entities"][0]["wkdata_qid"] == "Q130879"
    assert result["summary"] == [
        {
            "uid": "org.ent.pressagency.reuters",
            "wkdata_qid": "Q130879",
            "score": pytest.approx(result["entities"][0]["score"]),
        }
    ]


def test_min_score_filters_only_after_decoding_and_entity_scoring() -> None:
    pipe = make_pipeline(
        {},
        logits_by_token={
            "Reuters": [0.0, 0.3, 0.1, -4.0, -4.0],
        },
    )

    kept = pipe("Reuters", min_score=0.0, diagnostics=True)
    filtered = pipe("Reuters", min_score=0.99, diagnostics=True)

    assert kept["token_labels"] == ["B-org.ent.pressagency.reuters"]
    assert kept["entities"]
    assert filtered["token_labels"] == ["B-org.ent.pressagency.reuters"]
    assert filtered["entities"] == []
    assert filtered["summary"] == []


def test_long_input_uses_first_covering_window_for_overlaps() -> None:
    tokenizer = FakeTokenizer({"Reuters": 1})
    pipe = MediaSourcesPipeline(
        FakeModel(),
        tokenizer=tokenizer,
        device=-1,
        max_annotation_tokens=4,
        stride=2,
    )

    result = pipe("a b Reuters c d e", diagnostics=True)

    assert tokenizer.calls == [["a", "b", "Reuters", "c"], ["Reuters", "c", "d", "e"]]
    assert result["entities"][0]["surface"] == "Reuters"
    assert result["token_labels"] == ["O", "O", "B-org.ent.pressagency.reuters", "O", "O", "O"]


def test_predict_many_batches_windows_across_documents() -> None:
    model = FakeModel()
    tokenizer = FakeTokenizer({"Reuters": 1, "BBC": 3})
    pipe = MediaSourcesPipeline(model, tokenizer=tokenizer, device=-1, batch_size=2)

    results = pipe(["Reuters said.", "BBC said.", "No source."])

    assert model.batch_sizes == [2, 1]
    assert [call[0] for call in tokenizer.calls] == ["Reuters", "BBC", "No"]
    assert results[0]["entities"][0]["label"] == "org.ent.pressagency.reuters"
    assert results[1]["entities"][0]["label"] == "org.ent.radiostation.bbc"
    assert results[2]["entities"] == []


def test_predict_many_preserves_mixed_empty_document_order() -> None:
    pipe = make_pipeline({"Reuters": 1})

    results = pipe(["", "Reuters said.", "   "], diagnostics=True)

    assert len(results) == 3
    assert results[0] == {"entities": [], "summary": [], "text": ""}
    assert results[1]["entities"][0]["surface"] == "Reuters"
    assert results[2] == {"entities": [], "summary": [], "text": "   "}


def test_pipeline_rejects_incompatible_model_metadata() -> None:
    model = FakeModel()
    model.config.annotation_tokenization = "legacy"

    with pytest.raises(ValueError, match="unsupported annotation tokenization"):
        MediaSourcesPipeline(model, tokenizer=FakeTokenizer({}), device=-1)


def test_pipeline_drops_token_type_ids_for_modernbert_forward_signature() -> None:
    pipe = MediaSourcesPipeline(
        FakeModernBertModel(),
        tokenizer=TokenTypeIdsTokenizer({"Reuters": 1}),
        device=-1,
    )

    result = pipe("Reuters", diagnostics=True)

    assert result["entities"][0]["label"] == "org.ent.pressagency.reuters"


def test_unlinked_known_entity_gets_null_wkdata_qid() -> None:
    id2label = {
        0: "O",
        1: "B-org.ent.pressagency.agence-radio",
        2: "I-org.ent.pressagency.agence-radio",
    }
    model = FakeModel(id2label=id2label)
    tokenizer = FakeTokenizer({"Agence": 1, "Radio": 2})
    pipe = MediaSourcesPipeline(model, tokenizer=tokenizer, device=-1)

    result = pipe("Agence Radio")

    assert result["entities"] == [
        {
            "label": "org.ent.pressagency.agence-radio",
            "wkdata_qid": None,
            "start": 0,
            "stop": 12,
            "surface": "Agence Radio",
            "score": pytest.approx(1.0),
        }
    ]


def test_load_tokenizer_falls_back_for_tokenizers_backend_config(monkeypatch, tmp_path) -> None:
    tokenizer_config = tmp_path / "tokenizer_config.json"
    tokenizer_config.write_text(
        '{"tokenizer_class": "TokenizersBackend", "unk_token": "<unk>", "pad_token": "<pad>", "model_max_length": 8192}'
    )
    tokenizer_json = tmp_path / "tokenizer.json"
    tokenizer_json.write_text('{"version": "1.0"}')
    calls: list[tuple[str, str]] = []

    def fake_auto_from_pretrained(*_args, **_kwargs):
        raise ValueError("Tokenizer class TokenizersBackend does not exist or is not currently imported.")

    def fake_hf_hub_download(*, repo_id, filename, revision, local_files_only):
        calls.append((repo_id, filename))
        if filename == "tokenizer_config.json":
            return str(tokenizer_config)
        if filename == "tokenizer.json":
            return str(tokenizer_json)
        raise AssertionError(filename)

    class FakePreTrainedTokenizerFast:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(mediasources_module.AutoTokenizer, "from_pretrained", fake_auto_from_pretrained)
    monkeypatch.setattr(mediasources_module, "hf_hub_download", fake_hf_hub_download)
    monkeypatch.setattr(mediasources_module, "PreTrainedTokenizerFast", FakePreTrainedTokenizerFast)

    tokenizer = load_tokenizer("model-id", revision="v2.0.0", local_files_only=True, trust_remote_code=True)

    assert calls == [("model-id", "tokenizer_config.json"), ("model-id", "tokenizer.json")]
    assert tokenizer.kwargs == {
        "tokenizer_file": str(tokenizer_json),
        "unk_token": "<unk>",
        "pad_token": "<pad>",
        "model_max_length": 8192,
    }
