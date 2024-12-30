# import library
import numpy as np
import pandas as pd

from gensim.models import Word2Vec

from keras.preprocessing.sequence import pad_sequences 
from keras.models import Sequential 
from keras.layers import SimpleRNN, Dense, Embedding 


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer 

df = pd.read_csv("D:/for NLP/NLP-1/Derin_Öğrenme_Tabanlı_Modern_Dil_Modelleri/for.csv")

# metin verisi tokenize
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df["text"])
sequences = tokenizer.texts_to_sequences(df["text"])
word_index = tokenizer.word_index
print("Vocab size: ", len(word_index))

# padding process
maxlen = max(len(seq) for seq in sequences)
X = pad_sequences(sequences, maxlen=maxlen)
print("X shape: ",X.shape)

# label encoding
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df["label"])
print("Y shape: ", y.shape)

# train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 42)

# word embedding: word2vec, embedding matrisi olusturma
sentences = [text.split() for text in df["text"]]
word2vec_model = Word2Vec(sentences, vector_size=100, window=5, min_count=1)

embedding_dim = 100
embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
for word, i in word_index.items():
    if word in word2vec_model.wv:
        embedding_matrix[i] = word2vec_model.wv[word]
        
# built RNN model
model = Sequential()
model.add(Embedding(input_dim=len(word_index)+1, output_dim=embedding_dim, weights=[embedding_matrix], input_length=maxlen, trainable=False))
model.add(SimpleRNN(100, return_sequences=False))
model.add(Dense(1, activation="sigmoid"))

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# train RNN model
model.fit(X_train, y_train, epochs = 10, batch_size = 3, validation_data=(X_test, y_test))

# evaluate RNN model
print(" ")
loss, accuracy = model.evaluate(X_test, y_test)
print("Test loss: ", loss)
print("Test accuracy: ", accuracy)

# cumle sinifilandirma calismasi
def classify_sentence(sentence):
    
    seq = tokenizer.texts_to_sequences([sentence])
    padded_seq = pad_sequences(seq, maxlen=maxlen)
    
    prediction = model.predict(padded_seq)
    predicted_class = (prediction > 0.5).astype(int)
    label = "pozitif" if predicted_class[0][0] == 1 else "negatif"
    return label

sentence = "Otel çok temiz ve rahattı, çok keyif kaldık."

result = classify_sentence(sentence)
print("Tahmin: ",result)