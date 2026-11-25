"""
Deterministic, zero-dependency feature-hashing text embedder for reproducible
vector similarity benchmarks.

Uses SHA-256-seeded signed feature hashing over normalized word unigrams,
word bigrams, and character 4-grams so results are bit-for-bit reproducible
across Python versions, platforms, and process invocations without external
ML weights or network access.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import List

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "did",
        "do",
        "does",
        "for",
        "from",
        "has",
        "have",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "to",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "with",
    }
)


class DeterministicHashEmbedder:
    """Projects text into a normalized float vector using deterministic SHA-256 hashing."""

    def __init__(self, dim: int = 128, seed: int = 1347) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim
        self.seed = seed

    def _hash_feature(self, feature: str) -> tuple[int, float]:
        digest = hashlib.sha256(f"{self.seed}:{feature}".encode("utf-8")).digest()
        bucket = int.from_bytes(digest[:4], "big") % self.dim
        sign = 1.0 if (digest[4] & 1) == 0 else -1.0
        return bucket, sign

    def embed(self, text: str) -> List[float]:
        vec = [0.0] * self.dim
        raw_tokens = _TOKEN_RE.findall(text.lower())
        tokens = [t for t in raw_tokens if t not in _STOPWORDS] or raw_tokens
        if not tokens:
            vec[0] = 1.0
            return vec

        # Word unigrams (strong weight)
        for tok in tokens:
            idx, sign = self._hash_feature(f"w:{tok}")
            vec[idx] += sign * 2.0
            # Character 4-grams for morphological/stem overlap
            if len(tok) >= 4:
                for i in range(len(tok) - 3):
                    cg = tok[i : i + 4]
                    c_idx, c_sign = self._hash_feature(f"c:{cg}")
                    vec[c_idx] += c_sign * 0.45

        # Word bigrams (phrase specificity)
        for i in range(len(tokens) - 1):
            bg = f"{tokens[i]}_{tokens[i + 1]}"
            b_idx, b_sign = self._hash_feature(f"b:{bg}")
            vec[b_idx] += b_sign * 1.25

        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0.0:
            vec[0] = 1.0
            return vec
        return [v / norm for v in vec]
