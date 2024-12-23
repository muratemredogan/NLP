import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# veri seti yükle 
df = pd.read_csv("sms_spam.csv")

# tf-idf vektörleştirici
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["text"])

# kelime kümesi
feature_names = vectorizer.get_feature_names_out()
tfidf_score = X.mean(axis=0).A1 # ortalama tf-idf değerleri

df_tfidf = pd.DataFrame({"word": feature_names, "tfidf_score": tfidf_score})

df_tfidf_sorted = df_tfidf.sort_values(by="tfidf_score", ascending=False)
print(df_tfidf_sorted)