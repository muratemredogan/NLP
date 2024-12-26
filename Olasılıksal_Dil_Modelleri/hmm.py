import nltk
from nltk.tag import hmm

# örnek veri seti
train_data = [
    [("I","PRP"),("am","VBP"),("a","DT"),("student","NN")],
    [("You","PRP"),("am","VBP"),("a","DT"),("teacher","NN")]]

# hmm training
trainer = hmm.HiddenMarkovModelTrainer()
hmm_tagger = trainer.train(train_data)

# yeni cümle
test_sentence = "I am a teacher".split()
tags = hmm_tagger.tag(test_sentence)

print("etiketli cümle: ", tags)

"""
etiketli cümle:  [('I', 'PRP'), ('am', 'VBP'), ('a', 'DT'), ('teacher', 'NN')]
"""