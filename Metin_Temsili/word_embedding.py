"""
word embedding
    1) Word2Vec (Google)
    2) FastText (Facebook)
"""
import gensim
print(gensim.__version__)

import pandas as pd 
try:
    from gensim.models import Word2Vec, FastText
    from gensim.utils import simple_preprocess
except ImportError as e:
    print(f"Error importing gensim: {e}")
    print("Please ensure gensim is installed correctly.")
    exit(1)

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# örnek veri oluşturucam
sentences = [
    "Kedi çok tatlı bir hayvandır",
    "Köpekler evcil hayvanlardır.",
    "Kediler genellikle bağımsızdır.",
    "Köpekler sadık ve dost canlısıdır.",
    "Hayvanlar insanlar için iyi arkadaşlardır."
    ]

tokenized_sentences = [simple_preprocess(sentence) for sentence in sentences]

# word2vec
word2vec_model = Word2Vec(sentences=tokenized_sentences, vector_size=50, window=5, min_count=1, sg=0)

# fasttext
fasttext_model = FastText(sentences=tokenized_sentences, vector_size=50, window=5, min_count=1, sg=0)

def plot_word_embeddings(model,title):
    word_vectors = model.wv
    words = list(word_vectors.index_to_key)[:1000]
    vectors = [word_vectors[word] for word in words]

    # PCA
    pca = PCA(n_components=3)
    reduced_vectors = pca.fit_transform(vectors)

    # 3d gorselleştirme
    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(111, projection="3d")

    # vektörleri çizelim
    ax.scatter(reduced_vectors[:,0], reduced_vectors[:,1], reduced_vectors[:,2])

    # kelime etiketleri
    for i, word in enumerate(words):
        ax.text(reduced_vectors[i,0], reduced_vectors[i,1], reduced_vectors[i,2], word, fontsize=12)
    
    ax.set_title(title)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.set_zlabel("Component 3")
    plt.show()

plot_word_embeddings(word2vec_model, "Word2Vec")
plot_word_embeddings(fasttext_model, "FastText")