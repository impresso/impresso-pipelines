from __future__ import annotations

import random

import pytest

from impresso_pipelines.mediasources.decoding import (
    compile_bio_schema,
    decode_bio_viterbi_reference,
    viterbi_decode_with_schema,
)


def id2label(entity_count: int) -> dict[int, str]:
    labels = {0: "O"}
    for entity_index in range(entity_count):
        entity = f"ent.{entity_index}"
        labels[1 + entity_index * 2] = f"B-{entity}"
        labels[2 + entity_index * 2] = f"I-{entity}"
    return labels


@pytest.mark.parametrize(
    "emissions",
    [
        [[0.0, -1.0, -2.0]],
        [[0.0, 0.0, 0.0]],
        [[-10.0, 5.0, -10.0], [-10.0, -10.0, 5.0], [5.0, -10.0, -10.0]],
        [[-10.0, 5.0, -10.0], [-10.0, 5.0, -10.0]],
        [[0.0, 1.0, 1.0], [0.0, 1.0, 1.0], [1.0, 0.0, 0.0]],
    ],
)
def test_optimized_viterbi_matches_reference_for_edge_cases(emissions: list[list[float]]) -> None:
    labels = id2label(1)
    schema = compile_bio_schema(labels)

    assert viterbi_decode_with_schema(emissions, schema) == decode_bio_viterbi_reference(emissions, labels)


@pytest.mark.parametrize("entity_count", [1, 2, 8, 40])
@pytest.mark.parametrize("length", [1, 2, 7, 64])
def test_optimized_viterbi_matches_reference_for_random_emissions(entity_count: int, length: int) -> None:
    labels = id2label(entity_count)
    schema = compile_bio_schema(labels)
    rng = random.Random(entity_count * 10_000 + length)
    label_count = len(labels)
    emissions = [
        [rng.uniform(-12.0, 3.0) for _label_id in range(label_count)]
        for _position in range(length)
    ]

    assert viterbi_decode_with_schema(emissions, schema) == decode_bio_viterbi_reference(emissions, labels)


def test_optimized_viterbi_matches_reference_for_long_document() -> None:
    labels = id2label(12)
    schema = compile_bio_schema(labels)
    rng = random.Random(1948)
    label_count = len(labels)
    emissions = [
        [rng.uniform(-8.0, 4.0) for _label_id in range(label_count)]
        for _position in range(512)
    ]

    assert viterbi_decode_with_schema(emissions, schema) == decode_bio_viterbi_reference(emissions, labels)


def test_optimized_viterbi_matches_reference_for_more_than_byte_labels() -> None:
    labels = id2label(130)
    schema = compile_bio_schema(labels)
    rng = random.Random(256)
    label_count = len(labels)
    emissions = [
        [rng.uniform(-5.0, 5.0) for _label_id in range(label_count)]
        for _position in range(5)
    ]

    assert viterbi_decode_with_schema(emissions, schema) == decode_bio_viterbi_reference(emissions, labels)
