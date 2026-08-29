"""Character sets: the mapping between text and integer indices.

Small module, but it sits under three things that must agree with each other: the
fixture generator decides which characters to draw, the IAM parser decides which
transcriptions are usable, and the CER metric compares predicted text to target
text. If any of them disagrees about what "the alphabet" is, the disagreement
shows up as a quality problem rather than as an error, which is the worst way to
find a bug.

Design intent: an alphabet is *data*, not code. `register` adds one without
touching anything in this file, which is what keeps the door open to a second
script later at effectively zero cost.

Index layout is fixed and deliberate:

    0            pad   -- padding for variable-length batches
    1            unk   -- any character outside the alphabet
    2 .. 2+n-1   the alphabet, in the order given

pad is index 0 because that is what every masking and loss-ignore convention in
PyTorch assumes by default.
"""

from __future__ import annotations

from dataclasses import dataclass, field

PAD_INDEX = 0
UNK_INDEX = 1
_N_SPECIAL = 2

# The IAM alphabet: 79 characters. This is the set that appears in IAM
# transcriptions, and it is what the handwriting-recognition literature uses, so
# our CER numbers stay comparable to published ones.
_PUNCTUATION = " !\"#&'()*+,-./:;?"
_DIGITS = "0123456789"
_UPPERCASE = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_LOWERCASE = "abcdefghijklmnopqrstuvwxyz"

_REGISTRY: dict[str, str] = {
    "english": _PUNCTUATION + _DIGITS + _UPPERCASE + _LOWERCASE,
    "english_lower": _PUNCTUATION + _DIGITS + _LOWERCASE,
}


@dataclass(frozen=True)
class Charset:
    """An ordered alphabet plus its index mapping.

    Frozen because the mapping is baked into every artefact downstream -- a saved
    checkpoint, a packed dataset, a logged run. Changing it in place would silently
    invalidate all of them.
    """

    name: str
    characters: str
    _to_index: dict[str, int] = field(init=False, repr=False, compare=False)
    _to_char: dict[int, str] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.characters:
            raise ValueError(f"charset {self.name!r} is empty")

        duplicates = {c for c in self.characters if self.characters.count(c) > 1}
        if duplicates:
            raise ValueError(
                f"charset {self.name!r} has duplicate characters: {sorted(duplicates)!r}. "
                "Duplicates make encode/decode ambiguous."
            )

        to_index = {c: i + _N_SPECIAL for i, c in enumerate(self.characters)}
        object.__setattr__(self, "_to_index", to_index)
        object.__setattr__(self, "_to_char", {i: c for c, i in to_index.items()})

    def __len__(self) -> int:
        """Vocabulary size, including pad and unk. This is the number a model's
        output layer needs."""
        return len(self.characters) + _N_SPECIAL

    def __contains__(self, char: str) -> bool:
        return char in self._to_index

    def encode(self, text: str) -> list[int]:
        """Text to indices. Characters outside the alphabet become UNK_INDEX."""
        return [self._to_index.get(c, UNK_INDEX) for c in text]

    def decode(self, indices: list[int], keep_unknown: bool = False) -> str:
        """Indices back to text.

        pad is always dropped. unk is dropped too unless ``keep_unknown``, which
        renders it as a replacement character -- useful when eyeballing what a
        model actually produced.
        """
        out = []
        for i in indices:
            if i == PAD_INDEX:
                continue
            if i == UNK_INDEX:
                if keep_unknown:
                    out.append("\ufffd")
                continue
            char = self._to_char.get(int(i))
            if char is None:
                raise ValueError(f"index {i} is outside charset {self.name!r} (size {len(self)})")
            out.append(char)
        return "".join(out)

    def unsupported(self, text: str) -> set[str]:
        """Which characters in ``text`` this alphabet cannot represent.

        Used by the IAM parser to decide whether to keep a transcription, and to
        report *why* samples were dropped rather than dropping them silently.
        """
        return {c for c in text if c not in self._to_index}

    def supports(self, text: str) -> bool:
        return not self.unsupported(text)

    def filter(self, text: str) -> str:
        """Drop unsupported characters. Prefer rejecting a sample over filtering
        it: a filtered transcription no longer matches the image it labels."""
        return "".join(c for c in text if c in self._to_index)


def get(name: str) -> Charset:
    """Look up a registered alphabet by name. The name comes from `cfg.data.charset`."""
    if name not in _REGISTRY:
        raise KeyError(f"unknown charset {name!r}. Registered: {sorted(_REGISTRY)}")
    return Charset(name=name, characters=_REGISTRY[name])


def register(name: str, characters: str, overwrite: bool = False) -> Charset:
    """Add an alphabet at runtime.

    This exists so that adding a script is a data change rather than an
    architecture change. Nothing in this module needs editing to support one.
    """
    if name in _REGISTRY and not overwrite:
        raise KeyError(f"charset {name!r} already registered; pass overwrite=True")
    _REGISTRY[name] = characters
    return get(name)


def available() -> list[str]:
    return sorted(_REGISTRY)
