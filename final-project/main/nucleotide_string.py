from dataclasses import dataclass
from typing import Literal, NewType

Base3Int = NewType("Base3Int", str)


@dataclass(slots=True, frozen=True)
class NucleotideString:
    string: str  # nucleotide representation

    def __post_init__(self) -> None:
        # check nucleotide string is valid
        if not (set(self.string) <= {"A", "C", "G", "T"}):
            raise ValueError(f"Invalid Nucleotide Sequence: {self.string}")

    def __getitem__(self, index):
        if isinstance(index, slice):
            return NucleotideString(self.string[index.start : index.stop : index.step])
        elif isinstance(index, int):
            return NucleotideString(self.string[index])
        else:
            raise IndexError("Invalid index")

    def __add__(self, other):
        if isinstance(other, NucleotideString):
            return NucleotideString(self.string + other.string)
        raise ValueError(f"Cannot add f{type(other)} to NucleotideString")

    def __len__(self):
        return len(self.string)

    def __eq__(self, other):
        if not isinstance(other, NucleotideString):
            return NotImplemented
        return self.string == other.string

    def complement(self) -> "NucleotideString":
        """Return the complementary nucleotide string"""
        complementary_bases = {"A": "T", "C": "G", "G": "C", "T": "A"}
        return NucleotideString(
            "".join(list(map(lambda x: complementary_bases[x], list(self.string))))
        )

    def reverse(self) -> "NucleotideString":
        """Return the reversed nucleotide string"""
        return NucleotideString(self.string[::-1])

    def trits(self, start_row: Literal["A", "C", "G", "T"] = "A") -> Base3Int:
        """Returns the trits of the current sequence"""
        decoding_table = {
            ("A", "C"): "0",
            ("A", "G"): "1",
            ("A", "T"): "2",
            ("C", "G"): "0",
            ("C", "T"): "1",
            ("C", "A"): "2",
            ("G", "T"): "0",
            ("G", "A"): "1",
            ("G", "C"): "2",
            ("T", "A"): "0",
            ("T", "C"): "1",
            ("T", "G"): "2",
        }
        trits = ""
        for i, b in enumerate(list(self.string)):
            if i == 0:
                trits += decoding_table[(start_row, b)]
            else:
                trits += decoding_table[(self.string[i - 1], b)]
        return Base3Int(trits)

    @staticmethod
    def from_trits(
        trits: Base3Int, start_row: Literal["A", "C", "G", "T"] = "A"
    ) -> "NucleotideString":
        """returns nucleotide string for given trits"""
        encoding_table = {
            ("A", "0"): "C",
            ("A", "1"): "G",
            ("A", "2"): "T",
            ("C", "0"): "G",
            ("C", "1"): "T",
            ("C", "2"): "A",
            ("G", "0"): "T",
            ("G", "1"): "A",
            ("G", "2"): "C",
            ("T", "0"): "A",
            ("T", "1"): "C",
            ("T", "2"): "G",
        }
        nucleotide_string = ""
        for i, t in enumerate(list(trits)):
            if i == 0:
                nucleotide_string += encoding_table[(start_row, t)]
            else:
                nucleotide_string += encoding_table[(nucleotide_string[-1], t)]
        return NucleotideString(nucleotide_string)
