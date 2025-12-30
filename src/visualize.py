# Turns tensors into human-understandable pictures

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def visualize_codon_embeddings(model, codons):
    embedding_layer = model.get_layer("embedding")
    weights = embedding_layer.get_weights()[0]

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
