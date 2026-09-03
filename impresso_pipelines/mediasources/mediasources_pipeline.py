from __future__ import annotations

import json
import inspect
import logging
import re
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download
from transformers import AutoModelForTokenClassification, AutoTokenizer, PreTrainedTokenizerFast

from .decoding import (
    DECODER_CHOICES,
    DECODER_FIRST_SUBTOKEN_VITERBI,
    compile_bio_schema,
    decode_document,
    semantic_label_probability,
)
from .config import LABEL_START_YEARS, LABEL_WKDATA_QIDS


logger = logging.getLogger(__name__)


DEFAULT_MODEL = "impresso-project/mmbert-impresso-mediasources-ner"
DEFAULT_REVISION = "v2.0.0"
TOKENIZATION_PROFILE = "unicode-word-punctuation-v1"
DEFAULT_MAX_SEQUENCE_LEN = 512
DEFAULT_MAX_ANNOTATION_TOKENS = 256
DEFAULT_STRIDE = 48
TOKEN_RE = re.compile(r"[^\W\d_]+|\d+|_+|[^\w\s]", re.UNICODE)


@dataclass(frozen=True)
class Window:
    start_token: int
    tokens: list[str]


@dataclass(frozen=True)
class DocumentWindowingDiagnostics:
    annotation_tokens: int
    model_chunks: int
    overlapping_annotation_tokens: int
    overlap_replacements: int
    rescued_tokens: list[int]


@dataclass(frozen=True)
class WordLogProbResults:
    word_log_probs_by_doc: list[list[list[list[float]]]]
    diagnostics_by_doc: list[DocumentWindowingDiagnostics]
    stats: dict[str, Any]


def tokenize_with_offsets(text: str) -> tuple[list[str], list[int], list[int]]:
    matches = list(TOKEN_RE.finditer(text))
    return (
        [match.group(0) for match in matches],
        [match.start() for match in matches],
        [match.end() for match in matches],
    )


# Legacy annotation-word windowing protocol. Retained for training and
# compatibility experiments; tokenizer-native inference does not use this.
def make_windows(tokens: Sequence[str], *, max_annotation_tokens: int, stride: int) -> list[Window]:
    if max_annotation_tokens <= 0:
        raise ValueError("max_annotation_tokens must be positive")
    if stride < 0:
        raise ValueError("stride must not be negative")
    step = max_annotation_tokens - stride
    if step <= 0:
        raise ValueError("stride must be smaller than max_annotation_tokens")
    if not tokens:
        return []

    windows: list[Window] = []
    start = 0
    while start < len(tokens):
        stop = min(start + max_annotation_tokens, len(tokens))
        windows.append(Window(start_token=start, tokens=list(tokens[start:stop])))
        if stop == len(tokens):
            break
        start += step
    return windows


def normalize_id2label(id2label: dict[Any, str] | list[str]) -> dict[int, str]:
    if isinstance(id2label, list):
        return {index: str(label) for index, label in enumerate(id2label)}
    return {int(index): str(label) for index, label in id2label.items()}


def publication_year(publication_date: str | int | None) -> int | None:
    if publication_date is None:
        return None
    if isinstance(publication_date, int):
        return publication_date
    match = re.search(r"\d{4}", str(publication_date))
    return int(match.group(0)) if match else None


def load_tokenizer(
    model: str,
    *,
    revision: str | None,
    local_files_only: bool,
    trust_remote_code: bool,
) -> Any:
    try:
        return AutoTokenizer.from_pretrained(
            model,
            revision=revision,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
        )
    except ValueError as exc:
        if "Tokenizer class TokenizersBackend" not in str(exc):
            raise

    tokenizer_config_path = hf_hub_download(
        repo_id=model,
        filename="tokenizer_config.json",
        revision=revision,
        local_files_only=local_files_only,
    )
    tokenizer_path = hf_hub_download(
        repo_id=model,
        filename="tokenizer.json",
        revision=revision,
        local_files_only=local_files_only,
    )
    tokenizer_config = json.loads(Path(tokenizer_config_path).read_text())
    tokenizer_kwargs = {
        key: tokenizer_config[key]
        for key in (
            "bos_token",
            "eos_token",
            "unk_token",
            "sep_token",
            "pad_token",
            "cls_token",
            "mask_token",
            "model_max_length",
            "padding_side",
        )
        if key in tokenizer_config
    }
    return PreTrainedTokenizerFast(tokenizer_file=tokenizer_path, **tokenizer_kwargs)


def torch_dtype(dtype: str) -> Any:
    import torch

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    try:
        return dtype_map[dtype]
    except KeyError as exc:
        raise ValueError(f"unsupported dtype: {dtype!r}; expected one of {sorted(dtype_map)}") from exc


def bio_labels_to_entities(
    labels: Sequence[str],
    starts: Sequence[int],
    stops: Sequence[int],
    text: str,
    token_scores: Sequence[float],
    wkdata_qids: dict[str, str | None],
    label_start_years: dict[str, int],
    *,
    min_score: float | None = None,
    publication_year: int | None = None,
    filter_anachronistic: bool = False,
) -> list[dict[str, Any]]:
    entities: list[dict[str, Any]] = []
    active_start: int | None = None
    active_stop: int | None = None
    active_token_start: int | None = None
    active_token_stop: int | None = None
    active_label = ""

    def close() -> None:
        nonlocal active_start, active_stop, active_token_start, active_token_stop, active_label
        if active_start is None or active_stop is None or active_token_start is None or active_token_stop is None:
            return
        span_scores = token_scores[active_token_start:active_token_stop]
        score = float(sum(span_scores) / len(span_scores)) if span_scores else 0.0
        start_year = label_start_years.get(active_label)
        if filter_anachronistic and publication_year is not None and start_year is not None and start_year > publication_year:
            active_start = None
            active_stop = None
            active_token_start = None
            active_token_stop = None
            active_label = ""
            return
        if min_score is None or score >= min_score:
            entities.append(
                {
                    "surface": text[active_start:active_stop],
                    "label": active_label,
                    "wkdata_qid": wkdata_qids.get(active_label),
                    "start_year": start_year,
                    "start": active_start,
                    "stop": active_stop,
                    "score": score,
                }
            )
        active_start = None
        active_stop = None
        active_token_start = None
        active_token_stop = None
        active_label = ""

    for index, label in enumerate(labels):
        if label == "O":
            close()
            continue
        prefix, separator, entity_label = label.partition("-")
        if not separator or prefix not in {"B", "I"}:
            close()
            continue
        if prefix == "B" or active_start is None or active_label != entity_label:
            close()
            active_start = starts[index]
            active_token_start = index
            active_label = entity_label
        active_stop = stops[index]
        active_token_stop = index + 1
    close()
    return entities


class MediaSourcesPipeline:
    """Pipeline for joint Impresso media-source NER."""

    def __init__(
        self,
        model: str | Any = DEFAULT_MODEL,
        *,
        revision: str | None = DEFAULT_REVISION,
        tokenizer: Any | None = None,
        decoder: str | None = None,
        min_score: float | None = None,
        batch_size: int = 1,
        dtype: str = "float32",
        device: str | int | None = None,
        max_sequence_len: int | None = None,
        max_annotation_tokens: int | None = None,
        stride: int | None = None,
        filter_anachronistic: bool = False,
        local_files_only: bool = False,
        trust_remote_code: bool = True,
    ) -> None:
        self.model_id = model if isinstance(model, str) else None
        self.revision = revision
        self.default_min_score = min_score
        self.default_batch_size = batch_size
        self.default_filter_anachronistic = filter_anachronistic
        self.last_inference_stats: dict[str, Any] = {}
        self.device = self._resolve_device(device)

        if isinstance(model, str):
            self.model = AutoModelForTokenClassification.from_pretrained(
                model,
                revision=revision,
                dtype=torch_dtype(dtype),
                local_files_only=local_files_only,
                trust_remote_code=trust_remote_code,
            )
        else:
            self.model = model
        if tokenizer is not None:
            self.tokenizer = tokenizer
        elif isinstance(model, str):
            self.tokenizer = load_tokenizer(
                model,
                revision=revision,
                local_files_only=local_files_only,
                trust_remote_code=trust_remote_code,
            )
        else:
            raise ValueError("tokenizer must be provided when model is an instantiated model")

        self.id2label = normalize_id2label(self.model.config.id2label)
        self.schema = compile_bio_schema(self.id2label)
        self.wkdata_qids = dict(LABEL_WKDATA_QIDS)
        self.label_start_years = dict(LABEL_START_YEARS)
        self.model_forward_parameters = self._model_forward_parameters()
        self.decoder = decoder or self._model_protocol_value("subtoken_decoding", "decoder") or DECODER_FIRST_SUBTOKEN_VITERBI
        if self.decoder not in DECODER_CHOICES:
            raise ValueError(f"unsupported decoder: {self.decoder!r}; expected one of {DECODER_CHOICES}")

        self.max_sequence_len = (
            int(self._model_protocol_value("max_sequence_len") or DEFAULT_MAX_SEQUENCE_LEN)
            if max_sequence_len is None
            else max_sequence_len
        )
        self.max_annotation_tokens = (
            int(self._model_protocol_value("max_annotation_tokens", "max_words_per_window") or DEFAULT_MAX_ANNOTATION_TOKENS)
            if max_annotation_tokens is None
            else max_annotation_tokens
        )
        self.stride = (
            int(self._model_protocol_value("subtoken_stride", "stride_subtokens") or DEFAULT_STRIDE)
            if stride is None
            else stride
        )

        self._validate_model_protocol(decoder_override=decoder is not None)
        if self.max_sequence_len <= 0:
            raise ValueError("max_sequence_len must be positive")
        if self.max_annotation_tokens <= 0:
            raise ValueError("max_annotation_tokens must be positive")
        if self.stride < 0:
            raise ValueError("stride must not be negative")
        if self.stride >= self.max_sequence_len:
            raise ValueError("stride must be smaller than max_sequence_len")
        if hasattr(self.model, "to"):
            self.model.to(self.device)
        if hasattr(self.model, "eval"):
            self.model.eval()

    def _resolve_device(self, device: str | int | None) -> Any:
        import torch

        if device is None:
            if torch.cuda.is_available():
                return torch.device("cuda:0")
            if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        if device == -1:
            return torch.device("cpu")
        if isinstance(device, int):
            return torch.device(f"cuda:{device}")
        return torch.device(device)

    def _model_protocol_value(self, *names: str) -> Any:
        for name in names:
            value = getattr(self.model.config, name, None)
            if value is not None:
                return value
        return None

    def _model_forward_parameters(self) -> set[str] | None:
        forward = getattr(self.model, "forward", None)
        if forward is None:
            return None
        signature = inspect.signature(forward)
        for parameter in signature.parameters.values():
            if parameter.kind == inspect.Parameter.VAR_KEYWORD:
                return None
        return set(signature.parameters)

    def _validate_model_protocol(self, *, decoder_override: bool) -> None:
        annotation_tokenization = self._model_protocol_value("annotation_tokenization", "tokenization_profile")
        if annotation_tokenization is not None and annotation_tokenization != TOKENIZATION_PROFILE:
            raise ValueError(
                f"unsupported annotation tokenization: {annotation_tokenization!r}; expected {TOKENIZATION_PROFILE!r}"
            )
        declared_decoder = self._model_protocol_value("subtoken_decoding", "decoder")
        if declared_decoder is not None and declared_decoder not in DECODER_CHOICES:
            raise ValueError(f"unsupported model decoder: {declared_decoder!r}; expected one of {DECODER_CHOICES}")
        if declared_decoder is not None and self.decoder != declared_decoder and not decoder_override:
            raise ValueError(f"decoder mismatch: model declares {declared_decoder!r}, resolved {self.decoder!r}")

    def __call__(
        self,
        input_texts: str | Sequence[str],
        *,
        min_score: float | None = None,
        batch_size: int | None = None,
        publication_date: str | int | None = None,
        filter_anachronistic: bool | None = None,
        diagnostics: bool = False,
    ) -> dict[str, Any] | list[dict[str, Any]]:
        single = isinstance(input_texts, str)
        texts = [input_texts] if single else [str(text) for text in input_texts]
        outputs = self.predict_many(
            texts,
            min_score=min_score,
            batch_size=batch_size,
            publication_dates=publication_date,
            filter_anachronistic=filter_anachronistic,
            diagnostics=diagnostics,
        )
        return outputs[0] if single else outputs

    def predict_many(
        self,
        texts: Sequence[str],
        *,
        min_score: float | None = None,
        batch_size: int | None = None,
        publication_dates: str | int | Sequence[str | int | None] | None = None,
        filter_anachronistic: bool | None = None,
        diagnostics: bool = False,
    ) -> list[dict[str, Any]]:
        """Predict documents while batching model inference across annotation windows."""
        texts = [str(text) for text in texts]
        if not texts:
            self.last_inference_stats = {
                "documents": 0,
                "tokens": 0,
                "windows": 0,
                "model_batches": 0,
                "model_batch_windows": 0,
                "model_batch_sizes": [],
                "mean_windows_per_batch": 0.0,
                "batch_fill": 0.0,
                "tokenize_seconds": 0.0,
                "inference_seconds": 0.0,
                "model_forward_seconds": 0.0,
                "decode_seconds": 0.0,
                "pipeline_seconds": 0.0,
            }
            return []

        effective_min_score = self.default_min_score if min_score is None else min_score
        effective_batch_size = self.default_batch_size if batch_size is None else batch_size
        effective_filter_anachronistic = (
            self.default_filter_anachronistic if filter_anachronistic is None else filter_anachronistic
        )
        if effective_batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if isinstance(publication_dates, Sequence) and not isinstance(publication_dates, str):
            if len(publication_dates) != len(texts):
                raise ValueError("publication_dates must have the same length as texts")
            publication_years = [publication_year(date) for date in publication_dates]
        else:
            publication_years = [publication_year(publication_dates) for _text in texts]

        pipeline_started = time.perf_counter()
        tokenize_started = time.perf_counter()
        tokenized = [tokenize_with_offsets(text) for text in texts]
        tokenize_seconds = time.perf_counter() - tokenize_started
        tokens_by_doc = [tokens for tokens, _starts, _stops in tokenized]
        inference_started = time.perf_counter()
        word_log_prob_results = self._word_log_probs_many(
            tokens_by_doc,
            batch_size=effective_batch_size,
        )
        inference_seconds = time.perf_counter() - inference_started

        decode_started = time.perf_counter()
        results = [
            self._decode_result(
                text,
                tokens,
                starts,
                stops,
                word_log_probs,
                inference_diagnostics,
                publication_year_value,
                min_score=effective_min_score,
                filter_anachronistic=effective_filter_anachronistic,
                diagnostics=diagnostics,
            )
            for text, (tokens, starts, stops), word_log_probs, inference_diagnostics, publication_year_value in zip(
                texts,
                tokenized,
                word_log_prob_results.word_log_probs_by_doc,
                word_log_prob_results.diagnostics_by_doc,
                publication_years,
                strict=True,
            )
        ]
        decode_seconds = time.perf_counter() - decode_started
        pipeline_seconds = time.perf_counter() - pipeline_started
        self.last_inference_stats = {
            **word_log_prob_results.stats,
            "documents": len(texts),
            "tokens": sum(len(tokens) for tokens in tokens_by_doc),
            "tokenize_seconds": tokenize_seconds,
            "inference_seconds": inference_seconds,
            "decode_seconds": decode_seconds,
            "pipeline_seconds": pipeline_seconds,
        }
        self._log_entity_stats(results)
        return results

    def predict_one(
        self,
        text: str,
        *,
        min_score: float | None = None,
        publication_date: str | int | None = None,
        filter_anachronistic: bool | None = None,
        diagnostics: bool = False,
    ) -> dict[str, Any]:
        return self.predict_many(
            [text],
            min_score=min_score,
            batch_size=1,
            publication_dates=publication_date,
            filter_anachronistic=filter_anachronistic,
            diagnostics=diagnostics,
        )[0]

    def _log_entity_stats(self, results: Sequence[dict[str, Any]]) -> None:
        entity_counts = Counter(
            str(entity["label"])
            for result in results
            for entity in result.get("entities", [])
            if isinstance(entity, dict) and "label" in entity
        )
        entity_total = sum(entity_counts.values())
        documents_with_entities = sum(1 for result in results if result.get("entities"))
        if entity_counts:
            labels = ", ".join(f"{label}={count}" for label, count in entity_counts.most_common())
        else:
            labels = "none"
        logger.info(
            "MediaSources entities: documents=%d, documents_with_entities=%d, entities=%d, labels=%s",
            len(results),
            documents_with_entities,
            entity_total,
            labels,
        )

    def _decode_result(
        self,
        text: str,
        tokens: Sequence[str],
        starts: Sequence[int],
        stops: Sequence[int],
        word_log_probs: Sequence[Sequence[Sequence[float]]],
        inference_diagnostics: DocumentWindowingDiagnostics,
        publication_year_value: int | None,
        *,
        min_score: float | None,
        filter_anachronistic: bool,
        diagnostics: bool,
    ) -> dict[str, Any]:
        if not tokens:
            result: dict[str, Any] = {"entities": [], "summary": []}
            if diagnostics:
                result["text"] = text
                result["inference_diagnostics"] = {
                    "annotation_tokens": inference_diagnostics.annotation_tokens,
                    "model_chunks": inference_diagnostics.model_chunks,
                    "overlapping_annotation_tokens": inference_diagnostics.overlapping_annotation_tokens,
                    "overlap_replacements": inference_diagnostics.overlap_replacements,
                    "rescued_tokens": inference_diagnostics.rescued_tokens,
                }
            return result

        pred_ids = decode_document(word_log_probs, decoder=self.decoder, schema=self.schema)
        labels = [self.id2label[int(label_id)] for label_id in pred_ids]
        token_scores = [
            semantic_label_probability(subtokens[0], int(label_id), self.schema)
            for subtokens, label_id in zip(word_log_probs, pred_ids, strict=True)
        ]
        entities = bio_labels_to_entities(
            labels,
            starts,
            stops,
            text,
            token_scores,
            self.wkdata_qids,
            self.label_start_years,
            min_score=min_score,
            publication_year=publication_year_value,
            filter_anachronistic=filter_anachronistic,
        )
        summary = self._summary(entities)
        if diagnostics:
            return {
                "text": text,
                "tokens": list(tokens),
                "token_start_offsets": list(starts),
                "token_end_offsets": list(stops),
                "token_labels": labels,
                "token_scores": token_scores,
                "entities": entities,
                "summary": summary,
                "inference_diagnostics": {
                    "annotation_tokens": inference_diagnostics.annotation_tokens,
                    "model_chunks": inference_diagnostics.model_chunks,
                    "overlapping_annotation_tokens": inference_diagnostics.overlapping_annotation_tokens,
                    "overlap_replacements": inference_diagnostics.overlap_replacements,
                    "rescued_tokens": inference_diagnostics.rescued_tokens,
                },
            }
        return {"entities": entities, "summary": summary}

    def _summary(self, entities: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        best: dict[str, float] = {}
        for entity in entities:
            label = str(entity["label"])
            best[label] = max(best.get(label, 0.0), float(entity["score"]))
        return [
            {"uid": uid, "wkdata_qid": self.wkdata_qids.get(uid), "score": score}
            for uid, score in sorted(best.items(), key=lambda item: item[1], reverse=True)
        ]

    def _word_log_probs(self, tokens: Sequence[str]) -> list[list[list[float]]]:
        return self._word_log_probs_many([tokens], batch_size=1).word_log_probs_by_doc[0]

    def _word_log_probs_many(
        self,
        tokens_by_doc: Sequence[Sequence[str]],
        *,
        batch_size: int,
    ) -> WordLogProbResults:
        """Run tokenizer-native overflow chunks through the model in batches."""
        import torch

        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        word_log_probs: list[list[list[list[float]] | None]] = [
            [None for _ in tokens] for tokens in tokens_by_doc
        ]
        model_chunks_by_doc = [0 for _tokens in tokens_by_doc]
        overlapping_annotation_tokens_by_doc = [0 for _tokens in tokens_by_doc]
        overlap_replacements_by_doc = [0 for _tokens in tokens_by_doc]
        model_batches = 0
        model_batch_windows = 0
        model_forward_seconds = 0.0
        model_batch_sizes: list[int] = []

        def encode_documents(
            document_indexes: Sequence[int],
            token_batches: Sequence[Sequence[str]],
            base_token_indexes: Sequence[int],
        ) -> tuple[Any, list[int], list[int]]:
            encoding = self.tokenizer(
                [list(tokens) for tokens in token_batches],
                is_split_into_words=True,
                padding=True,
                truncation=True,
                max_length=self.max_sequence_len,
                stride=self.stride,
                return_overflowing_tokens=True,
                return_offsets_mapping=False,
                return_tensors="pt",
            )
            sample_mapping = encoding.get("overflow_to_sample_mapping")
            if sample_mapping is None:
                mapped_samples = list(range(len(token_batches)))
            elif hasattr(sample_mapping, "detach"):
                mapped_samples = [int(index) for index in sample_mapping.detach().cpu().tolist()]
            else:
                mapped_samples = [int(index) for index in sample_mapping]
            return (
                encoding,
                [document_indexes[sample_index] for sample_index in mapped_samples],
                [base_token_indexes[sample_index] for sample_index in mapped_samples],
            )

        def encoding_length(encoding: dict[str, Any], chunk_count: int) -> int:
            input_ids = encoding.get("input_ids")
            if hasattr(input_ids, "shape"):
                return int(input_ids.shape[0])
            return chunk_count

        def slice_encoding(encoding: dict[str, Any], start: int, stop: int, chunk_count: int) -> dict[str, Any]:
            sliced: dict[str, Any] = {}
            for key, value in encoding.items():
                if hasattr(value, "shape") and len(value.shape) > 0 and int(value.shape[0]) == chunk_count:
                    sliced[key] = value[start:stop]
                elif isinstance(value, list) and len(value) == chunk_count:
                    sliced[key] = value[start:stop]
                else:
                    sliced[key] = value
            return sliced

        def process_encoding(
            encoding: Any,
            doc_indexes_by_chunk: Sequence[int],
            base_token_indexes_by_chunk: Sequence[int],
            *,
            record_overflow: bool,
        ) -> None:
            nonlocal model_batches, model_batch_windows, model_forward_seconds
            chunk_count = len(doc_indexes_by_chunk)
            if record_overflow:
                chunks_by_doc: dict[int, list[int]] = {}
                for chunk_index, doc_index in enumerate(doc_indexes_by_chunk):
                    chunks_by_doc.setdefault(doc_index, []).append(chunk_index)
                for doc_index, chunk_indexes in chunks_by_doc.items():
                    model_chunks_by_doc[doc_index] += len(chunk_indexes)
                    if len(chunk_indexes) <= 1:
                        continue
                    for overflow_index, chunk_index in enumerate(chunk_indexes[1:], start=1):
                        word_ids = [
                            int(word_id)
                            for word_id in encoding.word_ids(batch_index=chunk_index)
                            if word_id is not None
                        ]
                        if not word_ids:
                            continue
                        first_token = base_token_indexes_by_chunk[chunk_index] + min(word_ids)
                        previous_word_ids = [
                            int(word_id)
                            for word_id in encoding.word_ids(batch_index=chunk_indexes[overflow_index - 1])
                            if word_id is not None
                        ]
                        previous_last_token = (
                            base_token_indexes_by_chunk[chunk_indexes[overflow_index - 1]]
                            + max(previous_word_ids, default=min(word_ids) - 1)
                        )
                        overlapping_annotation_tokens_by_doc[doc_index] += max(0, previous_last_token - first_token + 1)

            for batch_start in range(0, chunk_count, batch_size):
                batch_stop = min(batch_start + batch_size, chunk_count)
                actual_batch_size = batch_stop - batch_start
                model_inputs = self._model_inputs(
                    slice_encoding(dict(encoding), batch_start, batch_stop, encoding_length(encoding, chunk_count))
                )
                model_batches += 1
                model_batch_windows += actual_batch_size
                model_batch_sizes.append(actual_batch_size)
                model_started = time.perf_counter()
                outputs = self.model(**model_inputs)
                model_forward_seconds += time.perf_counter() - model_started
                log_probabilities = torch.log_softmax(outputs.logits.detach().cpu(), dim=-1)
                attention_mask = model_inputs.get("attention_mask")
                attention_values = attention_mask.detach().cpu().tolist() if attention_mask is not None else None

                for local_batch_index, chunk_index in enumerate(range(batch_start, batch_stop)):
                    doc_index = doc_indexes_by_chunk[chunk_index]
                    base_token_index = base_token_indexes_by_chunk[chunk_index]
                    word_ids = encoding.word_ids(batch_index=chunk_index)
                    chunk_word_log_probs: dict[int, list[list[float]]] = {}
                    for subtoken_index, word_id in enumerate(word_ids):
                        if attention_values is not None and not attention_values[local_batch_index][subtoken_index]:
                            continue
                        if word_id is None:
                            continue
                        absolute_token = base_token_index + int(word_id)
                        if not 0 <= absolute_token < len(tokens_by_doc[doc_index]):
                            continue
                        chunk_word_log_probs.setdefault(absolute_token, []).append(
                            log_probabilities[local_batch_index, subtoken_index].tolist()
                        )
                    for absolute_token, token_log_probs in chunk_word_log_probs.items():
                        existing_log_probs = word_log_probs[doc_index][absolute_token]
                        if existing_log_probs is not None and len(token_log_probs) > len(existing_log_probs):
                            overlap_replacements_by_doc[doc_index] += 1
                        if existing_log_probs is None or len(token_log_probs) > len(existing_log_probs):
                            word_log_probs[doc_index][absolute_token] = token_log_probs

        with torch.inference_mode():
            nonempty_doc_indexes = [doc_index for doc_index, tokens in enumerate(tokens_by_doc) if tokens]
            if nonempty_doc_indexes:
                encoding, doc_indexes_by_chunk, base_token_indexes_by_chunk = encode_documents(
                    nonempty_doc_indexes,
                    [tokens_by_doc[doc_index] for doc_index in nonempty_doc_indexes],
                    [0 for _doc_index in nonempty_doc_indexes],
                )
                process_encoding(encoding, doc_indexes_by_chunk, base_token_indexes_by_chunk, record_overflow=True)
            rescue_specs = [
                (doc_index, token_index, tokens[token_index])
                for doc_index, (tokens, doc_log_probs) in enumerate(zip(tokens_by_doc, word_log_probs, strict=True))
                for token_index, subtokens in enumerate(doc_log_probs)
                if not subtokens
            ]
            rescue_tokens_by_doc: list[list[int]] = [[] for _tokens in tokens_by_doc]
            for doc_index, token_index, _token in rescue_specs:
                rescue_tokens_by_doc[doc_index].append(token_index)
            if rescue_specs:
                rescue_encoding, rescue_doc_indexes_by_chunk, rescue_base_token_indexes_by_chunk = encode_documents(
                    [doc_index for doc_index, _token_index, _token in rescue_specs],
                    [[token] for _doc_index, _token_index, token in rescue_specs],
                    [token_index for _doc_index, token_index, _token in rescue_specs],
                )
                process_encoding(
                    rescue_encoding,
                    rescue_doc_indexes_by_chunk,
                    rescue_base_token_indexes_by_chunk,
                    record_overflow=False,
                )

        out: list[list[list[list[float]]]] = []
        for doc_index, (tokens, doc_log_probs) in enumerate(zip(tokens_by_doc, word_log_probs, strict=True)):
            doc_out: list[list[list[float]]] = []
            for token_index, subtokens in enumerate(doc_log_probs):
                if not subtokens:
                    raise ValueError(
                        "model/tokenizer produced no subtokens for "
                        f"document {doc_index}, token {token_index}: {tokens[token_index]!r}"
                    )
                doc_out.append(subtokens)
            out.append(doc_out)
        diagnostics_by_doc = [
            DocumentWindowingDiagnostics(
                annotation_tokens=len(tokens_by_doc[doc_index]),
                model_chunks=model_chunks_by_doc[doc_index],
                overlapping_annotation_tokens=overlapping_annotation_tokens_by_doc[doc_index],
                overlap_replacements=overlap_replacements_by_doc[doc_index],
                rescued_tokens=rescue_tokens_by_doc[doc_index],
            )
            for doc_index in range(len(tokens_by_doc))
        ]
        primary_windows = sum(model_chunks_by_doc)
        windows = model_batch_windows
        batch_fill = (model_batch_windows / (model_batches * batch_size)) if model_batches else 0.0
        stats = {
            "windows": windows,
            "primary_windows": primary_windows,
            "model_batches": model_batches,
            "model_batch_windows": model_batch_windows,
            "model_batch_sizes": model_batch_sizes,
            "mean_windows_per_batch": (model_batch_windows / model_batches) if model_batches else 0.0,
            "batch_fill": batch_fill,
            "model_forward_seconds": model_forward_seconds,
        }
        return WordLogProbResults(word_log_probs_by_doc=out, diagnostics_by_doc=diagnostics_by_doc, stats=stats)

    def _model_inputs(self, encoding: dict[str, Any]) -> dict[str, Any]:
        inputs = {}
        for key, value in encoding.items():
            if key in {"offset_mapping", "overflow_to_sample_mapping"}:
                continue
            if self.model_forward_parameters is not None and key not in self.model_forward_parameters:
                continue
            inputs[key] = value.to(self.device) if hasattr(value, "to") else value
        return inputs
