import nltk
from nltk.tag import hmm
from nltk.corpus import conll2000

# gerekli veri setini indir
nltk.download("conll2000")

# conll veri setini yükle
train_data = conll2000.tagged_sents("train.txt")
test_data = conll2000.tagged_sents("test.txt") # kendimizin test yapabilmesi içün

# hmm training
tariner = hmm.HiddenMarkovModelTrainer()
hmm_tagger = tariner.train(train_data)

# test
test_sentence = "I am not going to drive".split()
tags = hmm_tagger.tag(test_sentence)

print(tags)