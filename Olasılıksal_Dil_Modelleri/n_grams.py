from collections import Counter
import nltk
from nltk.util import ngrams
from nltk.tokenize import word_tokenize

# örnek veri setim
corpus = [
    "I love you",
    "I love you too",
    "I love programming",
    "I love apple",
    "You love me",
    "She loves apple",
    "I love you and you love me"
]

# tokenize etme
tokens = [word_tokenize(sentence.lower()) for sentence in corpus]

# n-gram -> n:2
bigrams = []
for token_list in tokens:
    bigrams.extend(list(ngrams(token_list, 2)))

# bigrams frekans counter
bigram_freq = Counter(bigrams)

# n-gram -> n:3
trigrams = []
for token_list in tokens:
    trigrams.extend(list(ngrams(token_list, 3)))

# trigrams frekans counter
trigram_freq = Counter(trigrams)

# "I love" bigramından sonra "you" veya "apple" gelme olasılıklarını hesaplama
bigram = ("i", "love")
prob_you = trigram_freq["i","love","you"] / bigram_freq[bigram]
prob_apple = trigram_freq["i","love","apple"] / bigram_freq[bigram]

print("you kelimesinin lma olasılığı: ", prob_you)
print("apple kelimesinin lma olasılığı: ", prob_apple)