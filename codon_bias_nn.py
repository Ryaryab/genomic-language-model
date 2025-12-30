import itertools
import numpy as np
import tensorflow as tf
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# Sanity ping so we know the file actually executed
# ---------------------------------------------------------------------
print(">>> FILE LOADED")


# ---------------------------------------------------------------------
# Codon utilities
# ---------------------------------------------------------------------
def build_codon_alphabet():
    """
    Generate all 64 DNA codons (AAA, AAC, ..., TTT).
    This defines the vocabulary for the entire model.
    """
    bases = ["A", "C", "G", "T"]
    codons = ["".join(c) for c in itertools.product(bases, repeat=3)]
    assert len(codons) == 64
    return codons


def make_organism_bias(seed: int, n_codons: int) -> np.ndarray:
    """
    Create a synthetic 'codon usage bias' for an organism.
    Think of this as an organism-specific accent.
    """
    rng = np.random.default_rng(seed)
    bias = rng.random(n_codons)
    bias /= bias.sum()  # turn into a proper probability distribution
    return bias


def generate_sequence(
    rng: np.random.Generator,
    bias: np.ndarray,
    length: int
) -> np.ndarray:
    """
    Sample a sequence of codon indices according to an organism's bias.
    Output is integer-encoded for the embedding layer.
    """
    return rng.choice(len(bias), size=length, p=bias)


# ---------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------
def make_dataset(
    organisms: dict,
    seq_len: int,
    seqs_per_org: int,
    seed: int = 123
):
    """
    Build a labeled dataset of codon sequences.

    Each organism contributes `seqs_per_org` sequences,
    each of length `seq_len`.
    """
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

    # Shuffle once at the end to avoid organism ordering effects
    idx = rng.permutation(len(X))
    return X[idx], y[idx], org_names


# ---------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------
def build_model(seq_len: int, n_classes: int):
    """
    Simple but principled architecture:

    - Embedding learns a continuous representation of codons
    - Global average pooling turns a sequence into statistics
    - Dense layers classify organism identity
    """
    print(">>> BUILDING MODEL")

    model = tf.keras.Sequential([
        # Embedding MUST be first in Sequential
        tf.keras.layers.Embedding(
            input_dim=64,          # number of codons
            output_dim=16,         # embedding dimensionality
            input_length=seq_len,  # length of codon sequence
            name="embedding"
        ),

        # Collapse sequence dimension -> one vector per sequence
        tf.keras.layers.GlobalAveragePooling1D(),

        # Small MLP head (no need to overdo it)
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(n_classes, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


# ---------------------------------------------------------------------
# Embedding visualization
# ---------------------------------------------------------------------
def visualize_codon_embeddings(model, codons):
    """
    Project the learned codon embeddings into 2D using PCA.
    Each point is a codon; distance reflects learned similarity.
    """
    embedding_layer = model.get_layer("embedding")
    weights = embedding_layer.get_weights()[0]  # shape: (64, embed_dim)

    pca = PCA(n_components=2)
    coords = pca.fit_transform(weights)

    plt.figure(figsize=(10, 10))
    for i, codon in enumerate(codons):
        x, y = coords[i]
        plt.scatter(x, y)
        plt.text(x + 0.02, y + 0.02, codon, fontsize=9)

    plt.title("Learned Codon Embedding Space (PCA)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(True)
    plt.show()


# ---------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------
def main():
    print(">>> ENTERED MAIN")

    codons = build_codon_alphabet()

    # Synthetic organisms with different codon preferences
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

    # Simple train/validation split
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

    # -----------------------------------------------------------------
    # Demo prediction to sanity-check behavior
    # -----------------------------------------------------------------
    rng = np.random.default_rng(999)
    demo_seq = generate_sequence(
        rng,
        organisms["org_B"],
        length=SEQ_LEN
    )[None, :]

    probs = model.predict(demo_seq, verbose=0)[0]
    pred = int(np.argmax(probs))

    print("\nDemo Prediction (Generated from Org_B):")
    for name, p in sorted(
        zip(org_names, probs),
        key=lambda t: -t[1]
    ):
        print(f" {name}: {p:.4f}")

    print(f"Predicted: {org_names[pred]}")

    # -----------------------------------------------------------------
    # Visualize what the model actually learned
    # -----------------------------------------------------------------
    visualize_codon_embeddings(model, codons)


# ---------------------------------------------------------------------
# Entry point (this MUST be top-level)
# ---------------------------------------------------------------------
if __name__ == "__main__":
    main()
