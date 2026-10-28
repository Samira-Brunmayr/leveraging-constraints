import numpy as np

from main.file_encoder import (
    Fragment,
    base3_encode,
    fragments_from_base3,
    get_bytes,
    unsafe_decode_fragments,
)
from main.nucleotide_string import Base3Int, NucleotideString

PATH_TO_CODE = "./codes/code.cd"


def test_base3_encode() -> None:
    bytes_ = bytearray("Birney and Goldman".encode("utf-8"))
    assert (
        base3_encode(bytes_, PATH_TO_CODE)
        == "20100202101010100021200012221110221201112000212210002212222212021100210122100110210111200021"
    )


def test_from_trits() -> None:
    assert (
        NucleotideString.from_trits(
            "20100202101010100021200012221110221201112000212210002212222212021100210122100110210111200021000000000000000000000000000010102",
            start_row="A",
        ).string
        == "TAGTATATCGACTAGTACAGCGTAGCATCTCGCAGCGAGATACGCTGCTACGCAGCATGCTGTGAGTATCGATGACGAGTGACTCTGTACAGTACGTACGTACGTACGTACGTACGTACGACTAT"
    )


def get_code() -> dict[int, Base3Int]:
    with open(PATH_TO_CODE, "r") as f:
        lines = f.readlines()
    split_lines = list(map(lambda line: line.split(" "), lines))
    code = {int(line[0]): Base3Int(line[1].strip()) for line in split_lines}
    return code


def test_code_satisfies_kraft() -> None:
    """Ensure code satisfies kraft inequality"""
    code = get_code()
    s = 0
    for v in code.values():
        s += 3 ** (-1 * len(v))
    assert s <= 1 and s >= 0


def test_code_unique_values() -> None:
    """Ensure code has unique values for each key"""
    code = get_code()
    assert len(code.values()) == len(set(code.values()))


def test_code_unique_keys() -> None:
    """Ensure there are 255 unique keys"""
    code = get_code()
    assert sorted(list(code.keys())) == list(range(0, 256))


def test_fragments_from_base3() -> None:
    """Test the encoding of data from bytes to DNA fragments"""
    bytes_ = bytearray("Birney and Goldman".encode("utf-8"))
    trits = base3_encode(bytes_, PATH_TO_CODE)
    fragment_strings = fragments_from_base3(trits, id_="12")
    f0 = fragment_strings[0].string
    f1 = fragment_strings[1].string
    # check correct body of encodings
    assert (
        f0[1:-1]
        == "TAGTATATCGACTAGTACAGCGTAGCATCTCGCAGCGAGATACGCTGCTACGCAGCATGCTGTGAGTATCGATGACGAGTGACTCTGTACAGTACGTACGATACGTACGTACGTC"
    )
    assert (
        f1[1:-1]
        == "ATAGTCGTACGTACGTACGTACGTACGTACGTACTGTACAGAGTCACTCGTCATCGATACTCACAGCATGCTGCGTAGCAGCGTATCTCGCTGCGAGATGATACGTACGTACGAG"
    )
    # ensure no homopolymers
    combined = f0 + f1
    for i, c in enumerate(list(combined)):
        if i == 0:
            continue
        assert c != combined[i - 1]


def test_fragment() -> None:
    """Test Fragment construction"""
    bytes_ = bytearray("Birney and Goldman".encode("utf-8"))
    trits = base3_encode(bytes_, PATH_TO_CODE)
    fragment_strings = fragments_from_base3(trits, id_="12")
    decoded_fragment_0 = Fragment.from_nucleotide_string(fragment_strings[0])
    decoded_fragment_1 = Fragment.from_nucleotide_string(fragment_strings[1])
    assert decoded_fragment_0.id_ == "12"
    assert decoded_fragment_0.i == 0
    assert decoded_fragment_1.id_ == "12"
    assert decoded_fragment_1.i == 1
    assert decoded_fragment_0.decoded_parity == decoded_fragment_0.parity
    assert decoded_fragment_1.decoded_parity == decoded_fragment_1.parity


def test_parity_change() -> None:
    """Ensure parity changes can be identified"""
    bytes_ = bytearray("Birney and Goldman".encode("utf-8"))
    trits = base3_encode(bytes_, PATH_TO_CODE)
    fragment_strings = fragments_from_base3(trits, id_="12")
    decoded_fragment_1 = Fragment.from_nucleotide_string(fragment_strings[1])
    corrupted_fragment_1 = Fragment(
        data=decoded_fragment_1.data,
        id_="20",
        i3="211110120212",
        decoded_parity=decoded_fragment_1.decoded_parity,
    )
    assert corrupted_fragment_1.parity != corrupted_fragment_1.decoded_parity


def test_unsafe_decode_fragments() -> None:
    """Test decoding of fragments assuming no data corruption"""

    bytes_ = bytearray("Birney and Goldman".encode("utf-8"))
    trits = base3_encode(bytes_, PATH_TO_CODE)
    fragment_strings = fragments_from_base3(trits, id_="12")
    fragments = list(
        map(lambda x: Fragment.from_nucleotide_string(x), fragment_strings)
    )
    np.random.shuffle(fragments)
    decoded_bytes_ = unsafe_decode_fragments(fragments, PATH_TO_CODE)
    assert bytes_ == decoded_bytes_
    assert decoded_bytes_.decode() == "Birney and Goldman"


def test_encoding_and_decoding_of_readme() -> None:
    """Check encoding -> decoding of larger file e.g. README.md"""
    bs = get_bytes("README.md")
    assert bs is not None
    x = fragments_from_base3(base3_encode(bs, PATH_TO_CODE))
    frags = []
    for i in x:
        frags.append(Fragment.from_nucleotide_string(i))
    decoded = unsafe_decode_fragments(frags, PATH_TO_CODE)
    assert decoded == bs
