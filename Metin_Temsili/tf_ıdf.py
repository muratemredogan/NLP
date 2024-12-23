from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np

#örnek oluşturulan belge
documents = [
    "Kedi çok tatlı bir hayvandır",
    "Kedi ve köpekler çok tatlı hayvanlardır",
    "Arılar bal üretirler"
    ]

tfidf_vectorizer = TfidfVectorizer()

# metinler -> sayısal
X = tfidf_vectorizer.fit_transform(documents)

# kelime kümesi
feature_names = tfidf_vectorizer.get_feature_names_out()

print("TF - IDF vektör temsilleri:")
vektor_temsili = X.toarray()
print(vektor_temsili)

df_tfidf = pd.DataFrame(vektor_temsili, columns=feature_names)

kedi_tfidf = df_tfidf["kedi"]
kedi_mean_tfidf = np.mean(kedi_tfidf)
print("Kedi kelimesinin ortalama TF-IDF değeri:", kedi_mean_tfidf)

