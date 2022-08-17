from transformers import DistilBertTokenizer, DistilBertModel
import torch

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertModel.from_pretrained("distilbert-base-uncased")

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

outputs = model(**inputs)

print(outputs)
last_hidden_states = outputs.last_hidden_state
#%%
print(inputs)
print(inputs.input_ids[0])
print(tokenizer.decode(inputs.input_ids[0]))
print(last_hidden_states.shape)