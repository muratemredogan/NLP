import spacy

nlp = spacy.load("en_core_web_sm")

# inclemek kelime
word = "my way is not findible"

# kelimeyi nlp işleminden geçir
doc = nlp(word)

for token in doc:

    print("Text: ", token.text)
    print("Lemma: ", token.lemma_)
    print("POS: ", token.pos_)
    print("Tag: ", token.tag_)
    print("Dependecy: ", token.dep_)
    print("Shape: ", token.shape_)
    print("Is Alpha: ", token.is_alpha)
    print("Is Stop: ", token.is_stop)
    print("Moorphology: ", token.morph)
    print(f"Is plural: {'Number=Plur' in token.morph}")
    print(" ")

