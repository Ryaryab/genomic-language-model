# The brain of the project.

import numpy as np

from dataset import (
    build_codon_alphabet,
    make_organism_bias,
    make_dataset,
    generate_sequence,
)
from model import build_model
from visualize import visualize_codon_embeddings


def main():
    print(">>> ENTERED MAIN")

    codons = build_codon_alphabet()

    organisms = {
        "org_A": make_organism_bias(seed=1, n_codons=len(codons)),
        "org_B": make_organism_bias(seed=2, n_codons=len(codons)),
        "org_C": make_organism_bias(seed=3, n_codons=len(codons)),
    }

    SEQ_LEN = 120
    SEQS_PER_ORG = 600

    X, y, org_names = make_dataset(
        organisms,
        seq_len=SEQ_LEN,
        seqs_per_org=SEQS_PER_ORG,
    )

    split = int(0.8 * len(X))
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]

    model = build_model(seq_len=SEQ_LEN, n_classes=len(org_names))
    model.summary()

    print(">>> STARTING TRAINING")

    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=10,
        batch_size=32,
        verbose=2,
    )

    rng = np.random.default_rng(999)
    demo_seq = generate_sequence(rng, organisms["org_B"], length=SEQ_LEN)[None, :]
    probs = model.predict(demo_seq, verbose=0)[0]
    pred = int(np.argmax(probs))

    print("\nDemo Prediction (Generated from Org_B):")
    for name, p in sorted(zip(org_names, probs), key=lambda t: -t[1]):
        print(f" {name}: {p:.4f}")
    print(f"Predicted: {org_names[pred]}")

    visualize_codon_embeddings(model, codons)


if __name__ == "__main__":
    main()
