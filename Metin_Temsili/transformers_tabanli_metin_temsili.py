from transformers import AutoTokenizer, AutoModel
import torch

# model ve tokenizer yukle
model_name = "bert-base-uncased" 
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# metni tanimla
text = "Transformers are amazing for natural language processing."

# metni tokenlara donustur
inputs = tokenizer(text, return_tensors = "pt")

# modeli kullanarak metin temsili olustur
with torch.no_grad():
    outputs = model(**inputs)
    
# cikislardan ilk tokenlari alalim
last_hidden_state = outputs.last_hidden_state
first_token_embedding = last_hidden_state[0,0,:].numpy()

print("Metin temsili: ilk token: ")
print(first_token_embedding)