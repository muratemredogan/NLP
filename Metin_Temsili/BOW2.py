import pandas as pd 
from sklearn.feature_extraction.text import CountVectorizer
import re
from collections import Counter
import nltk
from nltk.corpus import stopwords
nltk.download("stopwords")
# stop word liste yükleme
stop_words_eng = set(stopwords.words("english"))

# Çalışma dizinini ayarla
df = pd.read_csv("IMDB Dataset.csv")
df2 = df.head(100)

# metin verileri alalım
documents = df["review"]
labels = df["sentiment"] # pozitif veya negatif

# metin temizliği fonksiyonu
def clean_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text) # html tag temizliği
    text = re.sub(r'[^\w\s]', '', text) # noktalama işaretleri
    text = re.sub(r'\d', '', text) # rakamlar
    text = re.sub(r'\n', '', text) # satır sonu karakteri
    text = " ".join([word for word in text.split() if len(word) > 2]) # tek ve çift karakterler yani a ve an gibi
    text = " ".join([word for word in text.split() if word.lower() not in stop_words_eng]) # stopwords kaldırma
    return text

# metin temizleme
cleaned_documents = [clean_text(doc) for doc in documents]

# bow
vectorizer = CountVectorizer()

# metin -> sayısal vektör
X = vectorizer.fit_transform(cleaned_documents[:100])

# kelime kümesi
feature_names = vectorizer.get_feature_names_out()

# vvektör temsili:
print("Vektör temsili:")
vektor_temsili_2 = X.toarray()[:2]
print(vektor_temsili_2)

# vektör temsili dataframe
df_bow = pd.DataFrame(X.toarray(), columns=feature_names)

# kelime frekansi
word_counts = X.sum(axis=0).A1
word_freq = dict(zip(feature_names, word_counts))

#ilk 5 kelime
most_common_words = Counter(word_freq).most_common(5)
print("En çok geçen 5 kelime: ", most_common_words)