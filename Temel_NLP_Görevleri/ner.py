import pandas as pd
import spacy

nlp = spacy.load("en_core_web_sm")

content = "Reacher, Neagley, O’Donnell and Dixon discover that before his disappearance, fellow 110th member Tony Swan was exchanging cryptic emails about a federal contract called Little Wing. The team travels to Boston to ambush a senator’s aide with close ties to the project. But they wind up being ambushed themselves by a group looking to kill them."

doc = nlp(content)

for ent in doc.ents:
    print(ent.text, ent.start_char, ent.end_char, ent.label_)
    
entities = [(ent.text, ent.label_, ent.lemma_) for ent in doc.ents]
df = pd.DataFrame(entities, columns=["text", "type","lemma"])

print(df)