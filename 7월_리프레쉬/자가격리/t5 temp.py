from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch.optim import Adam

model = T5ForConditionalGeneration.from_pretrained("t5-small")
tokenizer = T5Tokenizer.from_pretrained('t5-small')
optimizer = Adam(model.parameters(), lr = 1e-5)

"""1"""
model.train()
task_prefix = "Predictions : banana's color is"
inputs = tokenizer(task_prefix, return_tensors="pt")
output = tokenizer("yellow color", return_tensors='pt')

optimizer.zero_grad()
output = model(input_ids=inputs.input_ids,
               attention_mask=inputs.attention_mask,
               labels=output.input_ids,
               decoder_attention_mask= output.attention_mask)

output.loss.backward()
optimizer.step()


"""2"""
optimizer.zero_grad()
task_prefix = "Predictions : apple's color is "
inputs = tokenizer(task_prefix, return_tensors="pt")
output = tokenizer("red color", return_tensors='pt')

output = model(input_ids=inputs.input_ids,
               attention_mask=inputs.attention_mask,
               labels=output.input_ids,
               decoder_attention_mask= output.attention_mask)


output.loss.backward()
optimizer.step()

"""3"""
optimizer.zero_grad()
task_prefix = "Predictions : melon's color is "
inputs = tokenizer(task_prefix, return_tensors="pt")
output = tokenizer("green color", return_tensors='pt')

output = model(input_ids=inputs.input_ids,
               attention_mask=inputs.attention_mask,
               labels=output.input_ids,
               decoder_attention_mask= output.attention_mask)


output.loss.backward()
optimizer.step()

"""test"""
model.eval()
test = "Predictions : kiwi's color is "
test_input = tokenizer(test, return_tensors='pt')

output_sequences = model.generate(**test_input)

print(output_sequences)

print(tokenizer.decode(output_sequences.squeeze(0)))


#%%
from transformers import RobertaTokenizer
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

print(tokenizer.encode_plus("Hello world!"))

