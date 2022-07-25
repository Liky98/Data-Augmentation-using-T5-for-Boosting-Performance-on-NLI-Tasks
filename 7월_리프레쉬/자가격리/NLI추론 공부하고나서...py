#%%
from transformers import AlbertForSequenceClassification, AlbertConfig, AlbertTokenizer

import SNLI_load_and_processing

from torch import nn
from torch.optim import Adam
from tqdm import tqdm
import torch

class ALBERT_base_classification(nn.Module):
    def __init__(self):
        super(ALBERT_base_classification, self).__init__()
        self.AlBERT_base_config = AlbertConfig(hidden_size=768,
                                              num_attention_heads=12,
                                              intermediate_size=3072,
                                              id2label={"0": "Yes",
                                                        "1": "?",
                                                        "2": "NO"}
                                              )

        self.model = AlbertForSequenceClassification(self.AlBERT_base_config)

    def forward(self, input_ids, token_type_ids, attention_mask, labels):
        output = self.model(input_ids = input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, labels=labels)
        return output

    def get_tokenizer(self):
        tokenizer= AlbertTokenizer.from_pretrained('albert-base-v2')
        return tokenizer

def test():
    model = ALBERT_base_classification()
    tokenizer = model.get_tokenizer()

    inputs = tokenizer("Hello, my dog is cute.[SEP] my cat is cute.", return_tensors="pt")
    print(inputs)
    with torch.no_grad():
        logits = model(**inputs).logits
    print(logits)
    predicted_class_id = logits.argmax().item()
    print(model.AlBERT_base_config.id2label[predicted_class_id])

def train(model,train_dataloader,val_dataloader):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    len_train_data = 550152
    len_val_data = 10000

    model.to(device)
    optimizer = Adam(model.parameters(), lr=1e-5)
    lossfunction = nn.CrossEntropyLoss()

    for epoch in range(2):
        model.train()
        total_acc_train = 0
        total_loss_train = 0
        train_dataloader = tqdm(train_dataloader, desc='Loading train dataset')
        for i, batch in enumerate(train_dataloader):


            optimizer.zero_grad()
            batch = {k: v.to(device) for k, v in batch.items()}

            output = model(batch['input_ids'],batch['token_type_ids'],batch['attention_mask'], batch['labels'])

            predict = torch.argmax(output.logits, dim=-1)

            batch_loss = lossfunction(output.logits, batch['labels'])

            total_loss_train = total_loss_train + batch_loss.item()
            acc = (predict == batch['labels']).sum().item()
            total_acc_train += acc

            output.loss.backward()
            optimizer.step()

            train_dataloader.set_description("Loss %.04f Acc %.04f | step %d" % (batch_loss, total_acc_train / len_train_data, i))

        total_acc_val = 0
        total_loss_val = 0
        with torch.no_grad():
            model.eval()
            for batch in tqdm(val_dataloader):

                output = model(**batch.to(device))

                predict = torch.argmax(output.logits, dim=-1)

                acc = (predict == batch['labels']).sum().item()

                batch_loss = lossfunction(output.logits, batch['labels'])

                total_loss_val += batch_loss.item()

                total_acc_val += acc



            print(f'Epochs: {epoch + 1} | Train Loss: {total_loss_train / len_train_data: .3f} \
                        | Train Accuracy: {total_acc_train / len_train_data: .3f} \
                        | Val Loss: {total_loss_val /len_val_data: .3f} \
                        | Val Accuracy: {total_acc_val / len_val_data: .3f}')

        return model


if __name__ == "__main__" :
    model = ALBERT_base_classification()
    tokenizer = model.get_tokenizer()

    train_dataloader, dev_dataloader, test_dataloader = SNLI_load_and_processing.snli_dataset(tokenizer)
    train(model=model, train_dataloader=train_dataloader, val_dataloader=dev_dataloader)