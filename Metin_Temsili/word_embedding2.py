# kutuphanelerin iceriye aktarilmasi
import pandas as pd
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess 
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import re

# veri setinin yuklenmesi
# metin verilerinin alinmasi
df = pd.read_csv(r"D:\for NLP\5_Metin Temsili\IMDB Dataset.csv")
print(df.head())

documents = df["review"]

# metin temizleme
def clean_text(text):
    text = text.lower()
    text = re.sub(r"\d+","", text) # sayilari temizle
    text = re.sub(r"[^\w\s]", "", text) # ozel karakterleri temizle
    text = " ".join([word for word in text.split() if len(word) > 2]) # kısa kelimeleri temizle
    return text
  
cleaned_documents = [clean_text(doc) for doc in documents]  

# cumleleri tokenization islemi
tokenized_documents = [simple_preprocess(doc) for doc in cleaned_documents]

# word2vec modeli tanimlayalim
model = Word2Vec(sentences=tokenized_documents, vector_size=50, window=5, min_count=1, sg=0)
word_vectors = model.wv

words = list(word_vectors.index_to_key)[:500]
vectors = [word_vectors[word] for word in words]

# clustering 2 veya 3 adet kume oluşturma
kmeans = KMeans(n_clusters = 2)
kmeans.fit(vectors)
clusters = kmeans.labels_ # 0,1,2

# pca 50 -> 2
pca = PCA(n_components=2)
reduced_vectors = pca.fit_transform(vectors)

# 2d gorsellestirme
plt.figure()
plt.scatter(reduced_vectors[:,0], reduced_vectors[:,1], c=clusters, cmap = "viridis")

centers = pca.transform(kmeans.cluster_centers_)
plt.scatter(centers[:,0], centers[:,1], c="red", marker="x", s = 120, label = "Merkez")
plt.legend()

# figure uzerine kelimeleri ekleme
for i, word in enumerate(words):
    plt.text(reduced_vectors[i,0], reduced_vectors[i,1], word, fontsize = 8)
    
plt.title("Word2Vec")
plt.show()