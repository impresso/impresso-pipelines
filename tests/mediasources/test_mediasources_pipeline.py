from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest
import torch

from impresso_pipelines.mediasources import mediasources_pipeline as mediasources_module
from impresso_pipelines.mediasources.config import LABEL_START_YEARS, LABEL_WKDATA_QIDS
from impresso_pipelines.mediasources.mediasources_pipeline import (
    MediaSourcesPipeline,
    load_tokenizer,
    tokenize_with_offsets,
    torch_dtype,
)
from impresso_pipelines.newsagencies import NewsAgenciesPipeline


ID2LABEL = {
    0: "O",
    1: "B-org.ent.pressagency.reuters",
    2: "I-org.ent.pressagency.reuters",
    3: "B-org.ent.radiostation.bbc",
    4: "I-org.ent.radiostation.bbc",
}

EXPECTED_MISSING_QIDS = {
    "org.ent.pressagency.agence-radio",
    "org.ent.pressagency.akp",
    "org.ent.pressagency.cip",
    "org.ent.pressagency.keystone",
    "org.ent.pressagency.kipa",
    "org.ent.pressagency.telegraphen-union",
}


class FakeEncoding(dict):
    def __init__(
        self,
        token_batches: list[list[str]],
        label_id_batches: list[list[int]],
        logits_by_token: list[list[list[float]]] | None = None,
        word_id_batches: list[list[int | None]] | None = None,
    ):
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
        self._word_ids = word_id_batches or [
            list(range(len(tokens))) + [None] * (max_len - len(tokens)) for tokens in token_batches
        ]
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


class TruncatingTokenizer(FakeTokenizer):
    def __init__(self, label_by_token: dict[str, int], *, covered_tokens: int):
        super().__init__(label_by_token)
        self.covered_tokens = covered_tokens

    def __call__(self, tokens: list[str] | list[list[str]], **_kwargs: object) -> FakeEncoding:
        token_batches = tokens if tokens and isinstance(tokens[0], list) else [tokens]
        self.calls.extend([list(token_batch) for token_batch in token_batches])
        represented_batches = [list(token_batch[: self.covered_tokens]) for token_batch in token_batches]
        label_id_batches = [
            [self.label_by_token.get(token, 0) for token in represented_batch]
            for represented_batch in represented_batches
        ]
        encoding = FakeEncoding(
            represented_batches,
            label_id_batches,
            word_id_batches=[
                list(range(len(represented_batch)))
                for represented_batch in represented_batches
            ],
        )
        encoding["_label_ids"] = encoding.label_ids
        return encoding


class OverflowTokenizer(FakeTokenizer):
    def __init__(self, label_by_token: dict[str, int]):
        super().__init__(label_by_token)
        self.chunks: list[list[str]] = []

    def __call__(self, tokens: list[str] | list[list[str]], **kwargs: object) -> FakeEncoding:
        token_batches = tokens if tokens and isinstance(tokens[0], list) else [tokens]
        self.calls.extend([list(token_batch) for token_batch in token_batches])
        max_length = int(kwargs["max_length"])
        stride = int(kwargs.get("stride", 0))
        step = max_length - stride
        chunk_batches: list[list[str]] = []
        word_id_batches: list[list[int | None]] = []
        sample_mapping: list[int] = []
        for sample_index, token_batch in enumerate(token_batches):
            start = 0
            while start < len(token_batch):
                stop = min(start + max_length, len(token_batch))
                chunk = list(token_batch[start:stop])
                chunk_batches.append(chunk)
                self.chunks.append(chunk)
                word_id_batches.append(list(range(start, stop)))
                sample_mapping.append(sample_index)
                if stop == len(token_batch):
                    break
                start += step
        label_id_batches = [
            [self.label_by_token.get(token, 0) for token in chunk]
            for chunk in chunk_batches
        ]
        encoding = FakeEncoding(chunk_batches, label_id_batches, word_id_batches=word_id_batches)
        encoding["_label_ids"] = encoding.label_ids
        encoding["overflow_to_sample_mapping"] = torch.tensor(sample_mapping)
        return encoding


class PartialBoundaryTokenizer(FakeTokenizer):
    def __call__(self, tokens: list[str] | list[list[str]], **_kwargs: object) -> FakeEncoding:
        token_batches = tokens if tokens and isinstance(tokens[0], list) else [tokens]
        self.calls.extend([list(token_batch) for token_batch in token_batches])
        encoding = FakeEncoding(
            [["alpha", "fragment"], ["fragment", "fragment", "fragment", "omega"]],
            [[0, 3], [1, 2, 2, 0]],
            word_id_batches=[[0, 1], [1, 1, 1, 2]],
        )
        encoding["_label_ids"] = encoding.label_ids
        encoding["overflow_to_sample_mapping"] = torch.tensor([0, 0])
        return encoding


class EqualBoundaryTokenizer(FakeTokenizer):
    def __call__(self, tokens: list[str] | list[list[str]], **_kwargs: object) -> FakeEncoding:
        token_batches = tokens if tokens and isinstance(tokens[0], list) else [tokens]
        self.calls.extend([list(token_batch) for token_batch in token_batches])
        encoding = FakeEncoding(
            [["alpha", "target"], ["target", "omega"]],
            [[0, 1], [3, 0]],
            word_id_batches=[[0, 1], [1, 2]],
        )
        encoding["_label_ids"] = encoding.label_ids
        encoding["overflow_to_sample_mapping"] = torch.tensor([0, 0])
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


def test_media_source_qid_and_start_year_mappings_have_same_labels() -> None:
    assert set(LABEL_START_YEARS) == set(LABEL_WKDATA_QIDS)


def test_all_media_source_labels_have_start_years() -> None:
    assert {label for label, year in LABEL_START_YEARS.items() if year is None} == set()


def test_missing_qids_are_explicitly_tracked() -> None:
    assert {label for label, qid in LABEL_WKDATA_QIDS.items() if qid is None} == EXPECTED_MISSING_QIDS


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
            "start_year": 1922,
            "start": 4,
            "stop": 21,
            "surface": "BBC World Service",
            "score": pytest.approx(1.0),
        }
    ]
    assert result["inference_diagnostics"] == {
        "annotation_tokens": 6,
        "model_chunks": 1,
        "overlapping_annotation_tokens": 0,
        "overlap_replacements": 0,
        "rescued_tokens": [],
    }


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
    assert result["entities"][0]["start_year"] == 1851
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


def test_anachronistic_filter_drops_entities_after_publication_year() -> None:
    pipe = make_pipeline({"Reuters": 1, "BBC": 3})

    result = pipe(
        ["Reuters said.", "BBC said."],
        publication_date=["1920-01-01", "1920-01-01"],
        filter_anachronistic=True,
        diagnostics=True,
    )

    assert result[0]["entities"][0]["label"] == "org.ent.pressagency.reuters"
    assert result[1]["entities"] == []
    assert result[1]["summary"] == []


def test_anachronistic_filter_is_inactive_without_publication_date() -> None:
    pipe = make_pipeline({"BBC": 3})

    result = pipe("BBC said.", filter_anachronistic=True)

    assert result["entities"][0]["label"] == "org.ent.radiostation.bbc"


def test_publication_dates_must_match_batch_length() -> None:
    pipe = make_pipeline({})

    with pytest.raises(ValueError, match="publication_dates must have the same length"):
        pipe.predict_many(["a", "b"], publication_dates=["1930"])


def test_default_stride_is_48_subtokens() -> None:
    pipe = make_pipeline({})

    assert pipe.stride == 48


def test_long_input_uses_first_covering_tokenizer_chunk_for_overlaps() -> None:
    tokenizer = OverflowTokenizer({"Reuters": 1})
    pipe = MediaSourcesPipeline(
        FakeModel(),
        tokenizer=tokenizer,
        device=-1,
        max_sequence_len=4,
        stride=2,
    )

    result = pipe("a b Reuters c d e", diagnostics=True)

    assert tokenizer.calls == [["a", "b", "Reuters", "c", "d", "e"]]
    assert tokenizer.chunks == [["a", "b", "Reuters", "c"], ["Reuters", "c", "d", "e"]]
    assert result["entities"][0]["surface"] == "Reuters"
    assert result["token_labels"] == ["O", "O", "B-org.ent.pressagency.reuters", "O", "O", "O"]


def test_truncated_window_tail_tokens_are_retried_as_singletons() -> None:
    tokenizer = TruncatingTokenizer({"49": 1}, covered_tokens=3)
    pipe = MediaSourcesPipeline(
        FakeModel(),
        tokenizer=tokenizer,
        device=-1,
        max_annotation_tokens=5,
        stride=1,
    )

    result = pipe("a b c 49 d", diagnostics=True)

    assert tokenizer.calls == [["a", "b", "c", "49", "d"], ["49"], ["d"]]
    assert result["token_labels"] == ["O", "O", "O", "B-org.ent.pressagency.reuters", "O"]
    assert result["entities"][0]["surface"] == "49"
    assert result["inference_diagnostics"] == {
        "annotation_tokens": 5,
        "model_chunks": 1,
        "overlapping_annotation_tokens": 0,
        "overlap_replacements": 0,
        "rescued_tokens": [3, 4],
    }


def test_tokenizer_native_overflow_uses_subtoken_stride_without_rescue() -> None:
    tokenizer = OverflowTokenizer({"ee": 1})
    pipe = MediaSourcesPipeline(
        FakeModel(),
        tokenizer=tokenizer,
        device=-1,
        max_sequence_len=5,
        stride=2,
    )

    result = pipe("aa bb cc dd ee ff gg hh ii jj kk ll", diagnostics=True)

    assert tokenizer.calls == [
        ["aa", "bb", "cc", "dd", "ee", "ff", "gg", "hh", "ii", "jj", "kk", "ll"],
    ]
    assert tokenizer.chunks == [
        ["aa", "bb", "cc", "dd", "ee"],
        ["dd", "ee", "ff", "gg", "hh"],
        ["gg", "hh", "ii", "jj", "kk"],
        ["jj", "kk", "ll"],
    ]
    assert result["entities"][0]["surface"] == "ee"
    assert result["inference_diagnostics"]["model_chunks"] == 4
    assert result["inference_diagnostics"]["overlapping_annotation_tokens"] == 6
    assert result["inference_diagnostics"]["rescued_tokens"] == []


def test_tokenizer_native_overflow_reconstructs_multiple_documents() -> None:
    tokenizer = OverflowTokenizer({"aa": 1, "gg": 3, "jj": 1})
    pipe = MediaSourcesPipeline(
        FakeModel(),
        tokenizer=tokenizer,
        device=-1,
        max_sequence_len=3,
        stride=1,
    )

    results = pipe(
        [
            "aa bb cc dd ee",
            "ff gg hh ii",
            "jj",
        ],
        diagnostics=True,
    )

    assert tokenizer.chunks == [
        ["aa", "bb", "cc"],
        ["cc", "dd", "ee"],
        ["ff", "gg", "hh"],
        ["hh", "ii"],
        ["jj"],
    ]
    assert [result["inference_diagnostics"]["model_chunks"] for result in results] == [2, 2, 1]
    assert [result["inference_diagnostics"]["overlapping_annotation_tokens"] for result in results] == [1, 1, 0]
    assert [result["token_labels"] for result in results] == [
        ["B-org.ent.pressagency.reuters", "O", "O", "O", "O"],
        ["O", "B-org.ent.radiostation.bbc", "O", "O"],
        ["B-org.ent.pressagency.reuters"],
    ]
    assert [result["inference_diagnostics"]["rescued_tokens"] for result in results] == [[], [], []]


def test_later_overlap_replaces_partial_word_representation() -> None:
    pipe = MediaSourcesPipeline(
        FakeModel(),
        tokenizer=PartialBoundaryTokenizer({}),
        device=-1,
    )

    word_log_probs = pipe._word_log_probs(["alpha", "fragment", "omega"])

    assert [len(subtokens) for subtokens in word_log_probs] == [1, 3, 1]


def test_later_overlap_replaces_partial_word_label() -> None:
    pipe = MediaSourcesPipeline(
        FakeModel(),
        tokenizer=PartialBoundaryTokenizer({}),
        device=-1,
    )

    result = pipe("alpha fragment omega", diagnostics=True)

    assert result["token_labels"] == ["O", "B-org.ent.pressagency.reuters", "O"]
    assert result["inference_diagnostics"]["overlap_replacements"] == 1


def test_equal_length_overlap_keeps_first_representation() -> None:
    pipe = MediaSourcesPipeline(
        FakeModel(),
        tokenizer=EqualBoundaryTokenizer({}),
        device=-1,
    )

    result = pipe("alpha target omega", diagnostics=True)

    assert result["token_labels"] == ["O", "B-org.ent.pressagency.reuters", "O"]
    assert result["inference_diagnostics"]["overlap_replacements"] == 0


def test_singleton_rescue_failure_still_raises_clear_error() -> None:
    tokenizer = TruncatingTokenizer({}, covered_tokens=0)
    pipe = MediaSourcesPipeline(
        FakeModel(),
        tokenizer=tokenizer,
        device=-1,
        max_annotation_tokens=2,
        stride=1,
    )

    with pytest.raises(
        ValueError,
        match="model/tokenizer produced no subtokens for document 0, token 0: 'a'",
    ):
        pipe("a")


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


def test_predict_many_logs_entity_statistics(caplog) -> None:
    pipe = make_pipeline({"Reuters": 1, "BBC": 3})

    with caplog.at_level(logging.INFO, logger=mediasources_module.__name__):
        pipe(["Reuters said.", "BBC said.", "No source."])

    assert (
        "MediaSources entities: documents=3, documents_with_entities=2, entities=2, "
        "labels=org.ent.pressagency.reuters=1, org.ent.radiostation.bbc=1"
    ) in caplog.text


def test_predict_many_preserves_mixed_empty_document_order() -> None:
    pipe = make_pipeline({"Reuters": 1})

    results = pipe(["", "Reuters said.", "   "], diagnostics=True)

    assert len(results) == 3
    assert results[0] == {
        "entities": [],
        "summary": [],
        "text": "",
        "inference_diagnostics": {
            "annotation_tokens": 0,
            "model_chunks": 0,
            "overlapping_annotation_tokens": 0,
            "overlap_replacements": 0,
            "rescued_tokens": [],
        },
    }
    assert results[1]["entities"][0]["surface"] == "Reuters"
    assert results[2] == {
        "entities": [],
        "summary": [],
        "text": "   ",
        "inference_diagnostics": {
            "annotation_tokens": 0,
            "model_chunks": 0,
            "overlapping_annotation_tokens": 0,
            "overlap_replacements": 0,
            "rescued_tokens": [],
        },
    }


def test_pipeline_rejects_incompatible_model_metadata() -> None:
    model = FakeModel()
    model.config.annotation_tokenization = "legacy"

    with pytest.raises(ValueError, match="unsupported annotation tokenization"):
        MediaSourcesPipeline(model, tokenizer=FakeTokenizer({}), device=-1)


def test_pipeline_requires_tokenizer_for_instantiated_model() -> None:
    with pytest.raises(ValueError, match="tokenizer must be provided"):
        MediaSourcesPipeline(FakeModel(), device=-1)


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
            "start_year": 1918,
            "start": 0,
            "stop": 12,
            "surface": "Agence Radio",
            "score": pytest.approx(1.0),
        }
    ]


def test_pipeline_passes_requested_dtype_to_model_loader(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    def fake_model_from_pretrained(*_args, **kwargs):
        calls.append(kwargs)
        return FakeModel()

    monkeypatch.setattr(mediasources_module.AutoModelForTokenClassification, "from_pretrained", fake_model_from_pretrained)
    monkeypatch.setattr(mediasources_module, "load_tokenizer", lambda *_args, **_kwargs: FakeTokenizer({}))

    MediaSourcesPipeline("model-id", revision="v2.0.0", dtype="float16", device=-1)

    assert calls[0]["dtype"] is torch.float16


def test_torch_dtype_rejects_unknown_dtype() -> None:
    with pytest.raises(ValueError, match="unsupported dtype"):
        torch_dtype("float64")


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
