import itertools
import numpy as np
import tensorflow as tf

print(">>> FILE LOADED")

def build_codon_alphabet():
    bases = ['A', 'C', 'G', 'T']
    codons = ["".join(c) for c in itertools.product(bases, repeat=3)]
    assert len(codons) == 64
    return codons

def make_organism_bias(seed: int, n_codons: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    bias = rng.random(n_codons)
    bias /= bias.sum()
    return bias

def generate_sequence(rng: np.random.Generator, bias: np.ndarray, length: int) -> np.ndarray:
    return rng.choice(len(bias), size=length, p=bias)

def make_dataset(organisms: dict, seq_len: int, seqs_per_org: int, seed: int = 123):
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

    idx = rng.permutation(len(X))
    return X[idx], y[idx], org_names

def build_model(seq_len: int, n_classes: int):
    print(">>> BUILDING MODEL")
    model = tf.keras.Sequential({
        tf.keras.layers.Input(shape=(seq_len,), dtype=tf.int32),
        tf.keras.layers.Embedding(input_dim=64, output_dim=16),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(n_classes, activation='softmax')
    })

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    return model

def main():
    print(">>> ENTERED MAIN")
    codons = build_codon_alphabet()

    organisms = {
        "org_A" : make_organism_bias(seed=1, n_codons=len(codons)),
        "org_B" : make_organism_bias(seed=2, n_codons=len(codons)),
        "org_C" : make_organism_bias(seed=3, n_codons=len(codons)),
    }

    SEQ_LEN = 120
    SEQS_PER_ORG = 600

    X, y, org_names = make_dataset(
        organisms,
        seq_len = SEQ_LEN,
        seqs_per_org=SEQS_PER_ORG
    )

    split = int(0.8 * len(X))
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]

    model = build_model(seq_len=SEQ_LEN, n_classes=len(org_names))
    model.summary()

    print(">>> STARTING TRAINING")

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=10,
        batch_size=32,
        verbose=2,
    )

    rng = np.random.default_rng(999)
    demo_seq = generate_sequence(rng, organisms["org_B"], length=SEQ_LEN)[None, :]
    probs = model.predict(demo_seq, verbose=0) [0]
    pred = int(np.argmax(probs))

    print("\nDemo Prediction (Generated from Org_B):")
    for name, p in sorted(zip(org_names,probs), key = lambda t: -t[1]):
        print(f" {name}:c{p:3f}")
    print(f"Predicted: {org_names [pred]}")

if __name__ == "__main__":
    main()

                          