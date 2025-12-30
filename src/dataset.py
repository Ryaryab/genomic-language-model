# All things biology-adjacent live here.

import itertools
import numpy as np


def build_codon_alphabet():
    # 64 possible codons: AAA → TTT
    bases = ["A", "C", "G", "T"]
    codons = ["".join(c) for c in itertools.product(bases, repeat=3)]
    assert len(codons) == 64
    return codons


def make_organism_bias(seed: int, n_codons: int) -> np.ndarray:
    # Synthetic codon usage bias for a fake organism
    rng = np.random.default_rng(seed)
    bias = rng.random(n_codons)
    bias /= bias.sum()
    return bias


def generate_sequence(rng: np.random.Generator, bias: np.ndarray, length: int) -> np.ndarray:
    # Sample codons according to organism-specific bias
    return rng.choice(len(bias), size=length, p=bias)


def make_dataset(
    organisms: dict,
    seq_len: int,
    seqs_per_org: int,
    seed: int = 123,
):
    rng = np.random.default_rng(seed)
    org_names = list(organisms.keys())

    X, y = [], []
    for label, name in enumerate(org_names):
        bias = organisms[name]
        for _ in range(seqs_per_org):
            X.append(generate_sequence(rng, bias, seq_len))
            y.append(label)

    X = np.array(X, dtype=np.int32)
    y = np.array(y, dtype=np.int32)

    # Shuffle once, globally
    idx = rng.permutation(len(X))
    return X[idx], y[idx], org_names
