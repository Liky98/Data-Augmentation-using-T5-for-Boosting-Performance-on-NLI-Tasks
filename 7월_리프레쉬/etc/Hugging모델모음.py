class Model_load :
    def albert_MaskedLM():
        from transformers import AlbertTokenizer, AlbertForMaskedLM
        tokenizer = AlbertTokenizer.from_pretrained("albert-base-v2")
        model = AlbertForMaskedLM.from_pretrained("albert-base-v2")
        return tokenizer, model

    def bert_MaskedLM():
        from transformers import BertTokenizer, BertForMaskedLM
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        model = BertForMaskedLM.from_pretrained("bert-base-uncased")
        return tokenizer, model

    def predict_MaskedLM(sentence, tokenizer, model):
        import torch
        # add mask_token
        inputs = tokenizer(sentence, return_tensors="pt")
        print(inputs)
        with torch.no_grad():
            logits = model(**inputs).logits

        # retrieve index of [MASK]
        mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
        predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)
        output = tokenizer.decode(predicted_token_id)
        return output



#%%
load = Model_load
tokenizer, model = load.bert_MaskedLM()
sentence = "Dog and [MASK] were same [MASK]."
output = load.predict_MaskedLM(sentence, tokenizer, model)

print(output)

#%%
import torch
from transformers import AlbertTokenizer, AlbertForMaskedLM

model = AlbertForMaskedLM.from_pretrained("albert-base-v2")
print(model)
