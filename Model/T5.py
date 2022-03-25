"""
T5모델 가져오는 코드
"""

# from transformers import T5Tokenizer, T5ForConditionalGeneration
#
# tokenizer = T5Tokenizer.from_pretrained("t5-11b")
# model = T5ForConditionalGeneration.from_pretrained("t5-11b")
#
# input_ids = tokenizer("The <extra_id_0> walks in <extra_id_1> park", return_tensors="pt").input_ids
# labels = tokenizer("<extra_id_0> cute dog <extra_id_1> the <extra_id_2>", return_tensors="pt").input_ids
# # the forward function automatically creates the correct decoder_input_ids
# loss = model(input_ids=input_ids, labels=labels).loss

#%%
from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")

# when generating, we will use the logits of right-most token to predict the next token
# so the padding should be on the left
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token  # to avoid an error

task_prefix = "translate English to Korean : "
sentences = ["The house is wonderful.", "I like to work in Lab."]  # use different length sentences to test batching
inputs = tokenizer([task_prefix + sentence for sentence in sentences], return_tensors="pt", padding=True)

output_sequences = model.generate(
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    do_sample=False  # disable sampling to test if batching affects output
)

print(tokenizer.batch_decode(output_sequences, skip_special_tokens=True))

## ['Das Haus ist wunderbar.', 'Ich arbeite gerne in NYC.']