from __future__ import annotations

import json
import inspect
import re
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
from .config import LABEL_WKDATA_QIDS


DEFAULT_MODEL = "impresso-project/mmbert-impresso-mediasources-ner"
DEFAULT_REVISION = "v2.0.0"
TOKENIZATION_PROFILE = "unicode-word-punctuation-v1"
DEFAULT_MAX_SEQUENCE_LEN = 512
DEFAULT_MAX_ANNOTATION_TOKENS = 256
DEFAULT_STRIDE = 32
TOKEN_RE = re.compile(r"[^\W\d_]+|\d+|_+|[^\w\s]", re.UNICODE)


@dataclass(frozen=True)
class Window:
    start_token: int
    tokens: list[str]


def tokenize_with_offsets(text: str) -> tuple[list[str], list[int], list[int]]:
    matches = list(TOKEN_RE.finditer(text))
    return (
        [match.group(0) for match in matches],
        [match.start() for match in matches],
        [match.end() for match in matches],
    )


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


def bio_labels_to_entities(
    labels: Sequence[str],
    starts: Sequence[int],
    stops: Sequence[int],
    text: str,
    token_scores: Sequence[float],
    wkdata_qids: dict[str, str | None],
    *,
    min_score: float | None = None,
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
        if min_score is None or score >= min_score:
            entities.append(
                {
                    "surface": text[active_start:active_stop],
                    "label": active_label,
                    "wkdata_qid": wkdata_qids.get(active_label),
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
        device: str | int | None = None,
        max_sequence_len: int | None = None,
        max_annotation_tokens: int | None = None,
        stride: int | None = None,
        local_files_only: bool = False,
        trust_remote_code: bool = True,
    ) -> None:
        self.model_id = model if isinstance(model, str) else None
        self.revision = revision
        self.default_min_score = min_score
        self.default_batch_size = batch_size
        self.device = self._resolve_device(device)

        if isinstance(model, str):
            self.model = AutoModelForTokenClassification.from_pretrained(
                model,
                revision=revision,
                local_files_only=local_files_only,
                trust_remote_code=trust_remote_code,
            )
        else:
            self.model = model
        self.tokenizer = tokenizer or load_tokenizer(
            model,
            revision=revision,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
        )

        self.id2label = normalize_id2label(self.model.config.id2label)
        self.schema = compile_bio_schema(self.id2label)
        self.wkdata_qids = dict(LABEL_WKDATA_QIDS)
        self.model_forward_parameters = self._model_forward_parameters()
        self.decoder = decoder or self._model_protocol_value("subtoken_decoding", "decoder") or DECODER_FIRST_SUBTOKEN_VITERBI
        if self.decoder not in DECODER_CHOICES:
            raise ValueError(f"unsupported decoder: {self.decoder!r}; expected one of {DECODER_CHOICES}")

        self.max_sequence_len = max_sequence_len or int(
            self._model_protocol_value("max_sequence_len") or DEFAULT_MAX_SEQUENCE_LEN
        )
        self.max_annotation_tokens = max_annotation_tokens or int(
            self._model_protocol_value("max_annotation_tokens", "max_words_per_window") or DEFAULT_MAX_ANNOTATION_TOKENS
        )
        self.stride = stride or int(self._model_protocol_value("stride", "stride_words") or DEFAULT_STRIDE)

        self._validate_model_protocol(decoder_override=decoder is not None)
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
        diagnostics: bool = False,
    ) -> dict[str, Any] | list[dict[str, Any]]:
        single = isinstance(input_texts, str)
        texts = [input_texts] if single else list(input_texts)
        effective_min_score = self.default_min_score if min_score is None else min_score
        _effective_batch_size = batch_size or self.default_batch_size
        outputs = [self.predict_one(str(text), min_score=effective_min_score, diagnostics=diagnostics) for text in texts]
        return outputs[0] if single else outputs

    def predict_one(self, text: str, *, min_score: float | None = None, diagnostics: bool = False) -> dict[str, Any]:
        tokens, starts, stops = tokenize_with_offsets(text)
        if not tokens:
            result: dict[str, Any] = {"entities": [], "summary": []}
            if diagnostics:
                result["text"] = text
            return result

        word_log_probs = self._word_log_probs(tokens)
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
            min_score=min_score,
        )
        summary = self._summary(entities)
        if diagnostics:
            return {
                "text": text,
                "tokens": tokens,
                "token_start_offsets": starts,
                "token_end_offsets": stops,
                "token_labels": labels,
                "token_scores": token_scores,
                "entities": entities,
                "summary": summary,
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
        import torch

        windows = make_windows(tokens, max_annotation_tokens=self.max_annotation_tokens, stride=self.stride)
        word_log_probs: list[list[list[float]] | None] = [None for _ in tokens]
        word_source_window: list[int | None] = [None for _ in tokens]

        with torch.no_grad():
            for window_index, window in enumerate(windows):
                encoding = self.tokenizer(
                    window.tokens,
                    is_split_into_words=True,
                    truncation=True,
                    max_length=self.max_sequence_len,
                    return_offsets_mapping=False,
                    return_tensors="pt",
                )
                word_ids = encoding.word_ids()
                model_inputs = self._model_inputs(dict(encoding))
                outputs = self.model(**model_inputs)
                log_probabilities = torch.log_softmax(outputs.logits.detach().cpu(), dim=-1)
                attention_mask = model_inputs.get("attention_mask")
                attention_values = attention_mask.detach().cpu().tolist()[0] if attention_mask is not None else None

                for subtoken_index, word_id in enumerate(word_ids):
                    if attention_values is not None and not attention_values[subtoken_index]:
                        continue
                    if word_id is None:
                        continue
                    absolute_token = window.start_token + int(word_id)
                    if not 0 <= absolute_token < len(tokens):
                        continue
                    source_window = word_source_window[absolute_token]
                    if source_window is None:
                        word_source_window[absolute_token] = window_index
                        word_log_probs[absolute_token] = []
                    if word_source_window[absolute_token] == window_index:
                        word_log_probs[absolute_token].append(log_probabilities[0, subtoken_index].tolist())

        out: list[list[list[float]]] = []
        for token_index, subtokens in enumerate(word_log_probs):
            if not subtokens:
                raise ValueError(f"model/tokenizer produced no subtokens for token {token_index}: {tokens[token_index]!r}")
            out.append(subtokens)
        return out

    def _model_inputs(self, encoding: dict[str, Any]) -> dict[str, Any]:
        inputs = {}
        for key, value in encoding.items():
            if key == "offset_mapping":
                continue
            if self.model_forward_parameters is not None and key not in self.model_forward_parameters:
                continue
            inputs[key] = value.to(self.device) if hasattr(value, "to") else value
        return inputs
