class Model_load :

    def __init__(self):
        self.x = None

    def albert_MaskedLM(self):
        from transformers import AlbertTokenizer, AlbertForMaskedLM
        tokenizer = AlbertTokenizer.from_pretrained("albert-base-v2")
        model = AlbertForMaskedLM.from_pretrained("albert-base-v2")
        return tokenizer, model

    def bert_MaskedLM(self):
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

    def mask_to_classification(self, model_name, len_label):
        from transformers import AutoModelForSequenceClassification
        from transformers import AutoConfig

        config_model = AutoConfig.from_pretrained(model_name)
        config_model.num_labels = len_label
        model = AutoModelForSequenceClassification.from_config(config_model)

        return model

    def predict_classification(self):

        return None

if __name__ == "__main__":
    load = Model_load
    tokenizer, model = load.bert_MaskedLM(None)
    sentence = "Dog and [MASK] were same [MASK]."
    a = tokenizer(sentence, return_tensors='pt')
    print(a)

    print(model(**a))


    #output = load.predict_MaskedLM(sentence, tokenizer, model)
    #print(output)

