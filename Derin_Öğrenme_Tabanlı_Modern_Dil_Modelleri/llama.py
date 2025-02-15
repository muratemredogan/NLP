from transformers import AutoTokenizer, AutoModelForCausalLM

# modeli ve tokenizer yukle
model_name = "huggyllama/llama-7b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# ornek baslangic cumlesi
text = "I go to swim for"

# tokenize
inputs = tokenizer.encode(text, return_tensors="pt")

# metin tamamlama
outputs = model.generate(inputs.input_ids, max_length=10)

# decode
genereted_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(genereted_text)