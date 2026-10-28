from dataclasses import dataclass

import numpy as np

from .nucleotide_string import Base3Int, NucleotideString


@dataclass(frozen=True, slots=True)
class Fragment:
    data: NucleotideString
    id_: Base3Int
    i3: Base3Int
    decoded_parity: Base3Int

    @property
    def i(self) -> int:
        return int(self.i3, 3)

    @property
    def parity(self) -> Base3Int:
        return Base3Int(
            str(
                (
                    int(self.id_[0])
                    + sum(
                        [
                            0 if i % 2 == 1 else int(trit)
                            for i, trit in enumerate(list(self.i3))
                        ]
                    )
                )
                % 3
            )
        )

    @staticmethod
    def from_nucleotide_string(frag: NucleotideString) -> "Fragment":
        """Given a nucleotide string, return a Fragment encompassing its
        properties"""
        # Strip the first and last bases
        if len(frag) != 117:
            raise ValueError("Fragment should be of length 117nt")

        frag = frag[1:-1]
        data_information = frag[:100]
        indexing_information = frag[100:].trits(start_row=data_information.string[-1])

        decoded_parity = Base3Int(indexing_information[14])
        id_ = Base3Int(indexing_information[:2])
        i3 = Base3Int(indexing_information[2:14])

        if int(i3, 3) % 2 != 0:
            data_information = data_information.reverse().complement()

        return Fragment(data_information, id_, i3, decoded_parity)


def fragments_from_base3(
    data: Base3Int, id_: Base3Int = Base3Int("00")
) -> list[NucleotideString]:
    """Base3Int data to nucleotide string as described by Goldman"""

    if len(id_) != 2:
        raise ValueError("ID must be 2 trits long")

    base3_len = Base3Int(np.base_repr(len(data), base=3))
    if len(base3_len) > 20:
        raise ValueError("Data too large")

    # compute s2 through s5
    s2 = Base3Int(((20 - len(base3_len)) * "0") + base3_len)
    s3 = Base3Int((25 - ((len(s2) + len(data)) % 25)) * "0")
    s4 = Base3Int(data + s3 + s2)
    s5 = NucleotideString.from_trits(s4, start_row="A")

    # compute fragments
    n_nucleotides = len(s5)
    nucleotide_fragments: list[NucleotideString] = []
    for i in range(0, (n_nucleotides // 25) - 3):
        i3 = Base3Int(np.base_repr(i, base=3))
        i3 = Base3Int(((12 - len(i3)) * "0") + i3)  # pad to length 12
        assert len(i3) == 12

        # compute parity trit (note that odd numbered trits are 1-indexed
        # - so in this case we take the even indices when indexing from 0)
        parity = Base3Int(
            str(
                (
                    int(id_[0])
                    + sum(
                        [
                            0 if i % 2 == 1 else int(trit)
                            for i, trit in enumerate(list(i3))
                        ]
                    )
                )
                % 3
            )
        )

        indexing_information = Base3Int(id_ + i3 + parity)
        if i % 2 == 0:
            fragment = s5[25 * i : 25 * i + 100]
        else:
            fragment = s5[25 * i : 25 * i + 100].reverse().complement()
        # append encoded indexing information
        fragment += NucleotideString.from_trits(
            indexing_information, start_row=fragment[-1].string
        )

        # prepend and append base
        if fragment[0].string == "A":
            fragment = NucleotideString("T") + fragment
        elif fragment[0].string == "T":
            fragment = NucleotideString("A") + fragment
        else:
            fragment = NucleotideString(np.random.choice(["A", "T"])) + fragment
        if fragment[-1].string == "C":
            fragment += NucleotideString("G")
        elif fragment[-1].string == "G":
            fragment += NucleotideString("C")
        else:
            fragment += NucleotideString(np.random.choice(["C", "G"]))
        nucleotide_fragments.append(fragment)
    return nucleotide_fragments


def get_bytes(file_path: str) -> bytearray | None:
    """Given file path, returns the binary represenation of file"""
    try:
        with open(file_path, "rb") as file:
            data = file.read()
        return bytearray(data)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    except IOError as e:
        print(f"Error reading file: {e}")
        return None


def naive_encode(bytes_: bytearray, path_to_code: str) -> list[NucleotideString]:
    with open(path_to_code, "r") as f:
        lines = f.readlines()
    split_lines = list(map(lambda line: line.split(" "), lines))
    code = {int(line[0]): str(line[1].strip()) for line in split_lines}
    full_str = "".join(list(map(lambda b: code[b], bytes_)))
    fragments = []
    for i in range(0, len(full_str), 120):
        fragments.append(NucleotideString(full_str[i : i + 120]))
    return fragments


def base3_encode(bytes_: bytearray, path_to_code: str) -> Base3Int:
    """Given the file path to a base 3 code and the bytes, encode the bytes"""
    with open(path_to_code, "r") as f:
        lines = f.readlines()
    split_lines = list(map(lambda line: line.split(" "), lines))
    code = {int(line[0]): Base3Int(line[1].strip()) for line in split_lines}
    return Base3Int("".join(list(map(lambda byte: code[byte], list(bytes_)))))


def base3_decode(path_to_code: str, trits: Base3Int) -> bytearray:
    """Given file path to base 3 code and string of trits, return bytearray"""
    with open(path_to_code, "r") as f:
        lines = f.readlines()
    split_lines = list(map(lambda line: line.split(" "), lines))
    code = {Base3Int(line[1].strip()): int(line[0]) for line in split_lines}
    bytes_ = []
    i = 0
    while len(trits) != 0 and i < len(trits):
        if trits[: i + 1] in code:
            bytes_.append(code[trits[: i + 1]])
            trits = Base3Int(trits[i + 1 :])
            i = 0
        i += 1
    if len(trits) != 0:
        raise ValueError("Invalid trits sequence for given code")
    return bytearray(bytes_)


def make_huff():
    """Simplify the provided huffman code"""
    with open("View_huff3.cd.new", "r", encoding="unicode_escape") as f:
        lines = f.readlines()
    split_lines = list(map(lambda line: line.split("\t"), lines))
    with open("code.cd", "w") as f:
        lines = list(map(lambda line: f"{line[2]} {line[3]}\n", split_lines))
        f.writelines(lines)


def unsafe_decode_fragments(fragments: list[Fragment], path_to_code: str) -> bytearray:
    """Given a list of fragments and a code, decodes them unsafely - assuming
    all fragments present and no data corruption or loss"""
    fragments = sorted(fragments, key=lambda x: x.i)
    full_nucleotide_sequence = ""
    for i, fragment in enumerate(fragments):
        if i == 0:
            full_nucleotide_sequence = fragment.data.string
        else:
            full_nucleotide_sequence += fragment.data.string[75:]
    trits = NucleotideString(full_nucleotide_sequence).trits("A")
    data_length = int(trits[-20:], 3)
    return base3_decode(path_to_code, trits[:data_length])


if __name__ == "__main__":
    bs = get_bytes("Doge.jpg")
    assert bs is not None
    fragments = fragments_from_base3(base3_encode(bs, "code.cd"))
    print(len(fragments))
