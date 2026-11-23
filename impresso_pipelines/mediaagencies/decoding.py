from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import dataclass


DECODER_FIRST_SUBTOKEN = "first_subtoken"
DECODER_FIRST_SUBTOKEN_VITERBI = "first_subtoken_viterbi"
DECODER_ALL_SUBTOKEN = "all_subtoken"
DECODER_ALL_SUBTOKEN_VITERBI = "all_subtoken_viterbi"
DECODER_CHOICES = (
    DECODER_FIRST_SUBTOKEN,
    DECODER_FIRST_SUBTOKEN_VITERBI,
    DECODER_ALL_SUBTOKEN,
    DECODER_ALL_SUBTOKEN_VITERBI,
)

NEG_INF = -math.inf


@dataclass(frozen=True)
class BioDecoderSchema:
    label_count: int
    o_id: int
    entity_names: tuple[str, ...]
    b_ids: tuple[int, ...]
    i_ids: tuple[int, ...]
    bio_pairs: tuple[tuple[int, int], ...]
    i_id_by_b_id: tuple[int, ...]
    is_b_by_id: tuple[bool, ...]
    is_i_by_id: tuple[bool, ...]
    prefix_by_id: tuple[str, ...]
    entity_index_by_id: tuple[int, ...]


def label_prefix(label: str) -> str:
    if label == "O":
        return "O"
    if label.startswith("B-"):
        return "B"
    if label.startswith("I-"):
        return "I"
    raise ValueError(f"unsupported BIO label: {label}")


def label_entity(label: str) -> str:
    if label == "O":
        return ""
    label_prefix(label)
    return label[2:]


def compile_bio_schema(id2label: dict[int, str]) -> BioDecoderSchema:
    ordered_ids = sorted(id2label)
    if ordered_ids != list(range(len(ordered_ids))):
        raise ValueError(f"decoder requires contiguous label IDs starting at 0: {ordered_ids}")

    o_ids = [label_id for label_id, label in id2label.items() if label == "O"]
    if len(o_ids) != 1:
        raise ValueError(f"decoder requires exactly one O label, found {len(o_ids)}")

    b_by_entity: dict[str, int] = {}
    i_by_entity: dict[str, int] = {}
    prefix_by_id = [""] * len(ordered_ids)
    entity_name_by_id = [""] * len(ordered_ids)
    for label_id in ordered_ids:
        label = id2label[label_id]
        prefix = label_prefix(label)
        prefix_by_id[label_id] = prefix
        entity_name = label_entity(label)
        entity_name_by_id[label_id] = entity_name
        if prefix == "B":
            if entity_name in b_by_entity:
                raise ValueError(f"duplicate B label for entity: {entity_name}")
            b_by_entity[entity_name] = label_id
        elif prefix == "I":
            if entity_name in i_by_entity:
                raise ValueError(f"duplicate I label for entity: {entity_name}")
            i_by_entity[entity_name] = label_id

    missing_i = sorted(set(b_by_entity) - set(i_by_entity))
    missing_b = sorted(set(i_by_entity) - set(b_by_entity))
    if missing_i:
        raise ValueError(f"decoder requires I labels for B entities: {missing_i}")
    if missing_b:
        raise ValueError(f"decoder requires B labels for I entities: {missing_b}")

    entity_names = tuple(sorted(b_by_entity, key=lambda entity: b_by_entity[entity]))
    entity_index_by_id = [-1] * len(ordered_ids)
    i_id_by_b_id = [-1] * len(ordered_ids)
    is_b_by_id = [False] * len(ordered_ids)
    is_i_by_id = [False] * len(ordered_ids)
    for entity_index, entity_name in enumerate(entity_names):
        b_id = b_by_entity[entity_name]
        i_id = i_by_entity[entity_name]
        entity_index_by_id[b_id] = entity_index
        entity_index_by_id[i_id] = entity_index
        i_id_by_b_id[b_id] = i_id
        is_b_by_id[b_id] = True
        is_i_by_id[i_id] = True

    return BioDecoderSchema(
        label_count=len(ordered_ids),
        o_id=o_ids[0],
        entity_names=entity_names,
        b_ids=tuple(b_by_entity[entity_name] for entity_name in entity_names),
        i_ids=tuple(i_by_entity[entity_name] for entity_name in entity_names),
        bio_pairs=tuple((b_by_entity[entity_name], i_by_entity[entity_name]) for entity_name in entity_names),
        i_id_by_b_id=tuple(i_id_by_b_id),
        is_b_by_id=tuple(is_b_by_id),
        is_i_by_id=tuple(is_i_by_id),
        prefix_by_id=tuple(prefix_by_id),
        entity_index_by_id=tuple(entity_index_by_id),
    )


def transition_score(previous_label: str, current_label: str) -> float:
    current_prefix = label_prefix(current_label)
    if current_prefix == "O":
        return 0.0

    current_entity = label_entity(current_label)
    if current_prefix == "I":
        previous_prefix = label_prefix(previous_label)
        if previous_prefix in {"B", "I"} and label_entity(previous_label) == current_entity:
            return 0.0
        return NEG_INF

    previous_prefix = label_prefix(previous_label)
    # Project-specific rule, not standard BIO: directly adjacent mentions with the
    # same entity label are treated as one mention, so B/I-X cannot transition to B-X.
    if previous_prefix in {"B", "I"} and label_entity(previous_label) == current_entity:
        return NEG_INF
    return 0.0


def start_score(label: str) -> float:
    return NEG_INF if label_prefix(label) == "I" else 0.0


def first_subtoken_emissions(word_subtoken_log_probs: Sequence[Sequence[Sequence[float]]]) -> list[list[float]]:
    emissions = []
    for subtokens in word_subtoken_log_probs:
        if not subtokens:
            raise ValueError("cannot decode word without subtoken probabilities")
        emissions.append([float(value) for value in subtokens[0]])
    return emissions


def all_subtoken_emissions(
    word_subtoken_log_probs: Sequence[Sequence[Sequence[float]]],
    schema: BioDecoderSchema,
) -> list[list[float]]:
    emissions: list[list[float]] = []
    for subtokens in word_subtoken_log_probs:
        if not subtokens:
            raise ValueError("cannot decode word without subtoken probabilities")
        word_emissions = [NEG_INF] * schema.label_count
        for label_id in range(schema.label_count):
            if label_id == schema.o_id:
                word_emissions[label_id] = float(sum(subtoken[schema.o_id] for subtoken in subtokens))
                continue
            if schema.is_b_by_id[label_id]:
                inside_id = schema.i_id_by_b_id[label_id]
                value = float(subtokens[0][label_id])
                value += float(sum(subtoken[inside_id] for subtoken in subtokens[1:]))
                word_emissions[label_id] = value
                continue
            if schema.is_i_by_id[label_id]:
                word_emissions[label_id] = float(sum(subtoken[label_id] for subtoken in subtokens))
        emissions.append(word_emissions)
    return emissions


def argmax_decode(emissions: Sequence[Sequence[float]]) -> list[int]:
    return [max(range(len(row)), key=lambda index: float(row[index])) for row in emissions]


def decode_bio_viterbi_reference(emissions: Sequence[Sequence[float]], id2label: dict[int, str]) -> list[int]:
    if not emissions:
        return []
    ordered_ids = sorted(id2label)
    compile_bio_schema(id2label)
    ordered_labels = [id2label[index] for index in ordered_ids]
    state_count = len(ordered_ids)
    scores = [float(emissions[0][state]) + start_score(ordered_labels[state]) for state in range(state_count)]
    backpointers: list[list[int]] = []
    for position in range(1, len(emissions)):
        next_scores: list[float] = []
        next_backpointers: list[int] = []
        for current in range(state_count):
            current_label = ordered_labels[current]
            best_previous = 0
            best_score = -math.inf
            for previous in range(state_count):
                score = scores[previous] + transition_score(ordered_labels[previous], current_label)
                if score > best_score:
                    best_score = score
                    best_previous = previous
            next_scores.append(best_score + float(emissions[position][current]))
            next_backpointers.append(best_previous)
        scores = next_scores
        backpointers.append(next_backpointers)

    best_state = max(range(state_count), key=lambda index: scores[index])
    states = [best_state]
    for pointers in reversed(backpointers):
        best_state = pointers[best_state]
        states.append(best_state)
    states.reverse()
    return [ordered_ids[state] for state in states]


def top_three_state_ids(scores: Sequence[float]) -> tuple[int, int, int]:
    # Ties intentionally keep the lowest state ID, matching the reference Viterbi
    # implementation's ascending predecessor scan and Python max() first-wins behavior.
    score_count = len(scores)
    if score_count < 3:
        raise ValueError("BIO top-three predecessor search requires at least three labels")
    id0 = 0
    id1 = 1
    id2 = 2
    score0 = scores[0]
    score1 = scores[1]
    score2 = scores[2]
    if score1 > score0:
        id0, id1 = id1, id0
        score0, score1 = score1, score0
    if score2 > score1:
        id1, id2 = id2, id1
        score1, score2 = score2, score1
        if score1 > score0:
            id0, id1 = id1, id0
            score0, score1 = score1, score0
    for state_id in range(3, score_count):
        score = scores[state_id]
        if score > score0:
            id2, score2 = id1, score1
            id1, score1 = id0, score0
            id0, score0 = state_id, score
        elif score > score1:
            id2, score2 = id1, score1
            id1, score1 = state_id, score
        elif score > score2:
            id2, score2 = state_id, score
    return id0, id1, id2


def best_predecessor_excluding_pair(
    top_three_ids: tuple[int, int, int],
    *,
    excluded_first: int,
    excluded_second: int,
) -> int:
    for state_id in top_three_ids:
        if state_id != excluded_first and state_id != excluded_second:
            return state_id
    raise ValueError("could not find legal predecessor outside excluded BIO pair")


def viterbi_decode_with_schema(
    emissions: Sequence[Sequence[float]],
    schema: BioDecoderSchema,
) -> list[int]:
    if not emissions:
        return []

    label_count = schema.label_count
    if label_count > 256:
        return _viterbi_decode_with_schema_list_backpointers(emissions, schema)

    o_id = schema.o_id
    bio_pairs = schema.bio_pairs
    is_i_by_id = schema.is_i_by_id
    previous_scores = [NEG_INF] * label_count
    current_scores = [NEG_INF] * label_count

    first_emissions = emissions[0]
    for label_id in range(label_count):
        if is_i_by_id[label_id]:
            previous_scores[label_id] = NEG_INF
        else:
            previous_scores[label_id] = float(first_emissions[label_id])

    emissions_count = len(emissions)
    backpointers = bytearray((emissions_count - 1) * label_count)
    for position in range(1, emissions_count):
        row = emissions[position]
        top0, top1, top2 = top_three_state_ids(previous_scores)
        backpointer_offset = (position - 1) * label_count

        current_scores[o_id] = float(row[o_id]) + previous_scores[top0]
        backpointers[backpointer_offset + o_id] = top0

        for b_id, i_id in bio_pairs:
            if top0 != b_id and top0 != i_id:
                b_predecessor = top0
            elif top1 != b_id and top1 != i_id:
                b_predecessor = top1
            else:
                b_predecessor = top2
            current_scores[b_id] = float(row[b_id]) + previous_scores[b_predecessor]
            backpointers[backpointer_offset + b_id] = b_predecessor

            b_score = previous_scores[b_id]
            i_score = previous_scores[i_id]
            if b_score > i_score:
                i_predecessor = b_id
                i_score_value = b_score
            elif i_score > b_score:
                i_predecessor = i_id
                i_score_value = i_score
            else:
                i_predecessor = b_id if b_id < i_id else i_id
                i_score_value = b_score
            current_scores[i_id] = float(row[i_id]) + i_score_value
            backpointers[backpointer_offset + i_id] = i_predecessor

        previous_scores, current_scores = current_scores, previous_scores

    best_state = max(range(label_count), key=lambda index: previous_scores[index])
    states = [0] * emissions_count
    states[-1] = best_state
    for position in range(emissions_count - 2, -1, -1):
        best_state = backpointers[position * label_count + best_state]
        states[position] = best_state
    return states


def _viterbi_decode_with_schema_list_backpointers(
    emissions: Sequence[Sequence[float]],
    schema: BioDecoderSchema,
) -> list[int]:
    if not emissions:
        return []
    label_count = schema.label_count
    o_id = schema.o_id
    bio_pairs = schema.bio_pairs
    is_i_by_id = schema.is_i_by_id
    previous_scores = [NEG_INF] * label_count
    current_scores = [NEG_INF] * label_count

    first_emissions = emissions[0]
    for label_id in range(label_count):
        if is_i_by_id[label_id]:
            previous_scores[label_id] = NEG_INF
        else:
            previous_scores[label_id] = float(first_emissions[label_id])

    emissions_count = len(emissions)
    backpointers = [0] * ((emissions_count - 1) * label_count)
    for position in range(1, emissions_count):
        row = emissions[position]
        top0, top1, top2 = top_three_state_ids(previous_scores)
        backpointer_offset = (position - 1) * label_count

        current_scores[o_id] = float(row[o_id]) + previous_scores[top0]
        backpointers[backpointer_offset + o_id] = top0

        for b_id, i_id in bio_pairs:
            if top0 != b_id and top0 != i_id:
                b_predecessor = top0
            elif top1 != b_id and top1 != i_id:
                b_predecessor = top1
            else:
                b_predecessor = top2
            current_scores[b_id] = float(row[b_id]) + previous_scores[b_predecessor]
            backpointers[backpointer_offset + b_id] = b_predecessor

            b_score = previous_scores[b_id]
            i_score = previous_scores[i_id]
            if b_score > i_score:
                i_predecessor = b_id
                i_score_value = b_score
            elif i_score > b_score:
                i_predecessor = i_id
                i_score_value = i_score
            else:
                i_predecessor = b_id if b_id < i_id else i_id
                i_score_value = b_score
            current_scores[i_id] = float(row[i_id]) + i_score_value
            backpointers[backpointer_offset + i_id] = i_predecessor

        previous_scores, current_scores = current_scores, previous_scores

    best_state = max(range(label_count), key=lambda index: previous_scores[index])
    states = [0] * emissions_count
    states[-1] = best_state
    for position in range(emissions_count - 2, -1, -1):
        best_state = backpointers[position * label_count + best_state]
        states[position] = best_state
    return states


def viterbi_decode(emissions: Sequence[Sequence[float]], id2label: dict[int, str]) -> list[int]:
    return viterbi_decode_with_schema(emissions, compile_bio_schema(id2label))


def viterbi_decode_first_subtoken_with_schema(
    word_subtoken_log_probs: Sequence[Sequence[Sequence[float]]],
    schema: BioDecoderSchema,
) -> list[int]:
    if not word_subtoken_log_probs:
        return []
    label_count = schema.label_count
    if label_count > 256:
        return _viterbi_decode_first_subtoken_with_schema_list_backpointers(word_subtoken_log_probs, schema)

    o_id = schema.o_id
    bio_pairs = schema.bio_pairs
    is_i_by_id = schema.is_i_by_id
    previous_scores = [NEG_INF] * label_count
    current_scores = [NEG_INF] * label_count

    first_subtokens = word_subtoken_log_probs[0]
    if not first_subtokens:
        raise ValueError("cannot decode word without subtoken probabilities")
    first_row = first_subtokens[0]
    for label_id in range(label_count):
        if is_i_by_id[label_id]:
            previous_scores[label_id] = NEG_INF
        else:
            previous_scores[label_id] = float(first_row[label_id])

    token_count = len(word_subtoken_log_probs)
    backpointers = bytearray((token_count - 1) * label_count)
    for position in range(1, token_count):
        subtokens = word_subtoken_log_probs[position]
        if not subtokens:
            raise ValueError("cannot decode word without subtoken probabilities")
        row = subtokens[0]
        top0, top1, top2 = top_three_state_ids(previous_scores)
        backpointer_offset = (position - 1) * label_count

        current_scores[o_id] = float(row[o_id]) + previous_scores[top0]
        backpointers[backpointer_offset + o_id] = top0

        for b_id, i_id in bio_pairs:
            if top0 != b_id and top0 != i_id:
                b_predecessor = top0
            elif top1 != b_id and top1 != i_id:
                b_predecessor = top1
            else:
                b_predecessor = top2
            current_scores[b_id] = float(row[b_id]) + previous_scores[b_predecessor]
            backpointers[backpointer_offset + b_id] = b_predecessor

            b_score = previous_scores[b_id]
            i_score = previous_scores[i_id]
            if b_score > i_score:
                i_predecessor = b_id
                i_score_value = b_score
            elif i_score > b_score:
                i_predecessor = i_id
                i_score_value = i_score
            else:
                i_predecessor = b_id if b_id < i_id else i_id
                i_score_value = b_score
            current_scores[i_id] = float(row[i_id]) + i_score_value
            backpointers[backpointer_offset + i_id] = i_predecessor

        previous_scores, current_scores = current_scores, previous_scores

    best_state = max(range(label_count), key=lambda index: previous_scores[index])
    states = [0] * token_count
    states[-1] = best_state
    for position in range(token_count - 2, -1, -1):
        best_state = backpointers[position * label_count + best_state]
        states[position] = best_state
    return states


def _viterbi_decode_first_subtoken_with_schema_list_backpointers(
    word_subtoken_log_probs: Sequence[Sequence[Sequence[float]]],
    schema: BioDecoderSchema,
) -> list[int]:
    if not word_subtoken_log_probs:
        return []
    label_count = schema.label_count
    o_id = schema.o_id
    bio_pairs = schema.bio_pairs
    is_i_by_id = schema.is_i_by_id
    previous_scores = [NEG_INF] * label_count
    current_scores = [NEG_INF] * label_count

    first_subtokens = word_subtoken_log_probs[0]
    if not first_subtokens:
        raise ValueError("cannot decode word without subtoken probabilities")
    first_row = first_subtokens[0]
    for label_id in range(label_count):
        if is_i_by_id[label_id]:
            previous_scores[label_id] = NEG_INF
        else:
            previous_scores[label_id] = float(first_row[label_id])

    token_count = len(word_subtoken_log_probs)
    backpointers = [0] * ((token_count - 1) * label_count)
    for position in range(1, token_count):
        subtokens = word_subtoken_log_probs[position]
        if not subtokens:
            raise ValueError("cannot decode word without subtoken probabilities")
        row = subtokens[0]
        top0, top1, top2 = top_three_state_ids(previous_scores)
        backpointer_offset = (position - 1) * label_count

        current_scores[o_id] = float(row[o_id]) + previous_scores[top0]
        backpointers[backpointer_offset + o_id] = top0

        for b_id, i_id in bio_pairs:
            if top0 != b_id and top0 != i_id:
                b_predecessor = top0
            elif top1 != b_id and top1 != i_id:
                b_predecessor = top1
            else:
                b_predecessor = top2
            current_scores[b_id] = float(row[b_id]) + previous_scores[b_predecessor]
            backpointers[backpointer_offset + b_id] = b_predecessor

            b_score = previous_scores[b_id]
            i_score = previous_scores[i_id]
            if b_score > i_score:
                i_predecessor = b_id
                i_score_value = b_score
            elif i_score > b_score:
                i_predecessor = i_id
                i_score_value = i_score
            else:
                i_predecessor = b_id if b_id < i_id else i_id
                i_score_value = b_score
            current_scores[i_id] = float(row[i_id]) + i_score_value
            backpointers[backpointer_offset + i_id] = i_predecessor

        previous_scores, current_scores = current_scores, previous_scores

    best_state = max(range(label_count), key=lambda index: previous_scores[index])
    states = [0] * token_count
    states[-1] = best_state
    for position in range(token_count - 2, -1, -1):
        best_state = backpointers[position * label_count + best_state]
        states[position] = best_state
    return states


def decode_document(
    word_subtoken_log_probs: Sequence[Sequence[Sequence[float]]],
    *,
    decoder: str,
    id2label: dict[int, str] | None = None,
    schema: BioDecoderSchema | None = None,
) -> list[int]:
    if decoder == DECODER_FIRST_SUBTOKEN:
        return argmax_decode(first_subtoken_emissions(word_subtoken_log_probs))
    if schema is None:
        if id2label is None:
            raise ValueError("all-subtoken decoding requires id2label or schema")
        schema = compile_bio_schema(id2label)
    if decoder == DECODER_FIRST_SUBTOKEN_VITERBI:
        return viterbi_decode_first_subtoken_with_schema(word_subtoken_log_probs, schema)
    if decoder == DECODER_ALL_SUBTOKEN:
        return argmax_decode(all_subtoken_emissions(word_subtoken_log_probs, schema))
    if decoder == DECODER_ALL_SUBTOKEN_VITERBI:
        emissions = all_subtoken_emissions(word_subtoken_log_probs, schema)
        return viterbi_decode_with_schema(emissions, schema)
    raise ValueError(f"unsupported decoder: {decoder}")


def decode_document_timed(
    word_subtoken_log_probs: Sequence[Sequence[Sequence[float]]],
    *,
    decoder: str,
    id2label: dict[int, str] | None = None,
    schema: BioDecoderSchema | None = None,
) -> tuple[list[int], float]:
    if decoder == DECODER_FIRST_SUBTOKEN:
        return argmax_decode(first_subtoken_emissions(word_subtoken_log_probs)), 0.0
    if schema is None:
        if id2label is None:
            raise ValueError("all-subtoken decoding requires id2label or schema")
        schema = compile_bio_schema(id2label)
    if decoder == DECODER_FIRST_SUBTOKEN_VITERBI:
        started = time.perf_counter()
        result = viterbi_decode_first_subtoken_with_schema(word_subtoken_log_probs, schema)
        return result, time.perf_counter() - started
    if decoder == DECODER_ALL_SUBTOKEN:
        return argmax_decode(all_subtoken_emissions(word_subtoken_log_probs, schema)), 0.0
    if decoder == DECODER_ALL_SUBTOKEN_VITERBI:
        emissions = all_subtoken_emissions(word_subtoken_log_probs, schema)
        started = time.perf_counter()
        result = viterbi_decode_with_schema(emissions, schema)
        return result, time.perf_counter() - started
    raise ValueError(f"unsupported decoder: {decoder}")


def semantic_label_probability(log_probabilities: Sequence[float], label_id: int, schema: BioDecoderSchema) -> float:
    """Probability of the decoded semantic label, independent of B/I structure."""
    probabilities = [math.exp(float(value)) for value in log_probabilities]
    if label_id == schema.o_id:
        return probabilities[schema.o_id]
    entity_index = schema.entity_index_by_id[label_id]
    if entity_index < 0:
        raise ValueError(f"label ID is not part of a BIO entity: {label_id}")
    return probabilities[schema.b_ids[entity_index]] + probabilities[schema.i_ids[entity_index]]


def semantic_label_margin(log_probabilities: Sequence[float], label_id: int, schema: BioDecoderSchema) -> float:
    """Margin between the decoded semantic label bucket and the strongest alternative."""
    probabilities = [math.exp(float(value)) for value in log_probabilities]
    buckets = [probabilities[schema.o_id]]
    buckets.extend(
        probabilities[b_id] + probabilities[i_id]
        for b_id, i_id in zip(schema.b_ids, schema.i_ids, strict=True)
    )
    selected_bucket = 0 if label_id == schema.o_id else schema.entity_index_by_id[label_id] + 1
    selected_probability = buckets[selected_bucket]
    competitor = max((value for index, value in enumerate(buckets) if index != selected_bucket), default=0.0)
    return selected_probability - competitor
