"""Toy obfuscation used to tag warehouse labels. Not cryptography."""

KEY = "orbit"


def encode(word):
    out = []
    for i, ch in enumerate(word):
        shift = ord(KEY[i % len(KEY)]) - ord("a")
        out.append(chr((ord(ch) - ord("a") + shift + i) % 26 + ord("a")))
    return "".join(reversed(out))
