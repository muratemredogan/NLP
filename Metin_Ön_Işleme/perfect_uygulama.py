import nltk
import string
import re
from textblob import TextBlob
from bs4 import BeautifulSoup
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords

nltk.data.path.append("D:\\nltk_data")  # Kendi nltk_data dizin yolunuzu girin.

# Gerekli kaynakları indiriyor
nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

print("Tüm kaynaklar başarıyla indirildi!")

# Başlangıç metni
text = "Hello, world! In 2024, artificial intelligence is reshaping technology; let's embrace it with love and passion."

# Fazla boşlukları kaldırma
cleaned_text1 = " ".join(text.split())
print("Boşluklardan arındırılmış metin:", cleaned_text1)

# Büyük harften küçük harfe çevirme
cleaned_text2 = cleaned_text1.lower()
print("Küçük harfe çevrilmiş metin:", cleaned_text2)

# Noktalama işaretlerini kaldırma
cleaned_text3 = cleaned_text2.translate(str.maketrans("", "", string.punctuation))
print("Noktalama işaretlerinden arındırılmış metin:", cleaned_text3)

# Özel karakterleri kaldırma
cleaned_text4 = re.sub(r"[^A-Za-z0-9\s]", "", cleaned_text3)
print("Özel karakterlerden arındırılmış metin:", cleaned_text4)

# Yazım hatalarını düzeltme
cleaned_text5 = str(TextBlob(cleaned_text4).correct())
print("Yazım hataları düzeltilmiş metin:", cleaned_text5)

# HTML veya URL etiketlerini kaldırma
cleaned_text6 = BeautifulSoup(cleaned_text5, "html.parser").get_text()
print("HTML etiketlerinden arındırılmış metin:", cleaned_text6)

# Kelimeleri tokenlerine ayırma
word_tokens = nltk.word_tokenize(cleaned_text6)
print("Word Tokens:")
for word in word_tokens:
    print(word)

# Cümle tokenizasyonu
sentence_tokens = nltk.sent_tokenize(cleaned_text6)
print("\nSentence Tokens:")
for sentence in sentence_tokens:
    print(sentence)

# Stemming
stemmer = PorterStemmer()
stems = [stemmer.stem(w) for w in word_tokens]
print("Stems:", stems)

# Lemmatization
lemmatizer = WordNetLemmatizer()
lemmas_v = [lemmatizer.lemmatize(w, pos="v") for w in word_tokens]  # Lemmatize as verbs
lemmas_n = [lemmatizer.lemmatize(w, pos="n") for w in word_tokens]  # Lemmatize as nouns
print("Lemmas (verbs):", lemmas_v)
print("Lemmas (nouns):", lemmas_n)

# Stop words çıkarma
stop_words_eng = set(stopwords.words("english"))
filtered_words = [word for word in cleaned_text6.split() if word.lower() not in stop_words_eng]
print("Stop words çıkarılmış kelimeler:", filtered_words)
