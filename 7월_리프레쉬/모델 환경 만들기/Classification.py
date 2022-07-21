from transformers import BertTokenizer
from Hugging_GLUE import GLUE_dataset
from Hugging모델모음 import BertClassifier
from torch.optim import Adam
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader

labels = {'contradiction' : 0,
         'Entailment' : 1}

def dataset_batch(datset_name) :
    glue = GLUE_dataset
    raw_dataset = glue.load_GLUE_from_Huggingface(datset_name)

    return raw_dataset

def decode_sentence(input) :
    example_text = tokenizer.decode(input.input_ids[0])
    return example_text

def encode(examples):
    return tokenizer(examples['premise'], examples['hypothesis'], truncation=True, padding='max_length')


def train(model, train_data, val_data, learning_rate, epochs):

    train, val = Dataset(train_data), Dataset(val_data)

    train_dataloader = DataLoader(train, batch_size=2, shuffle=True)
    val_dataloader = DataLoader(val, batch_size=2)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    if use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    for epoch_num in range(epochs):

        total_acc_train = 0
        total_loss_train = 0

        for train_input, train_label in tqdm(train_dataloader):
            train_label = train_label.to(device)
            mask = train_input['attention_mask'].to(device)
            input_id = train_input['input_ids'].squeeze(1).to(device)

            output = model(input_id, mask)

            batch_loss = criterion(output, train_label.long())
            total_loss_train += batch_loss.item()

            acc = (output.argmax(dim=1) == train_label).sum().item()
            total_acc_train += acc

            model.zero_grad()
            batch_loss.backward()
            optimizer.step()

        total_acc_val = 0
        total_loss_val = 0

        with torch.no_grad():

            for val_input, val_label in val_dataloader:
                val_label = val_label.to(device)
                mask = val_input['attention_mask'].to(device)
                input_id = val_input['input_ids'].squeeze(1).to(device)

                output = model(input_id, mask)

                batch_loss = criterion(output, val_label.long())
                total_loss_val += batch_loss.item()

                acc = (output.argmax(dim=1) == val_label).sum().item()
                total_acc_val += acc

        print(
            f'Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(train_data): .3f} \
                | Train Accuracy: {total_acc_train / len(train_data): .3f} \
                | Val Loss: {total_loss_val / len(val_data): .3f} \
                | Val Accuracy: {total_acc_val / len(val_data): .3f}')


def evaluate(model, test_data):
    test = Dataset(test_data)

    test_dataloader = torch.utils.data.DataLoader(test, batch_size=2)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:
        model = model.cuda()

    total_acc_test = 0
    with torch.no_grad():

        for test_input, test_label in test_dataloader:
            test_label = test_label.to(device)
            mask = test_input['attention_mask'].to(device)
            input_id = test_input['input_ids'].squeeze(1).to(device)

            output = model(input_id, mask)

            acc = (output.argmax(dim=1) == test_label).sum().item()
            total_acc_test += acc

    print(f'Test Accuracy: {total_acc_test / len(test_data): .3f}')


if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

    # example_text = 'I want to go home'
    # bert_input = tokenizer(example_text,padding='max_length', max_length = 256,
    #                        truncation=True, return_tensors="pt")
    #
    # print(bert_input['input_ids'])
    # print(bert_input['token_type_ids'])
    # print(bert_input['attention_mask'])
    # print(decode_sentence(bert_input))

    dataset = dataset_batch('mnli')
    print(dataset['train'])

    dataset = dataset.map(encode, batched=True)

    print(dataset)

    EPOCHS = 5
    model = BertClassifier()
    LR = 1e-6

    train(model, dataset['train'], dataset['validation_matched'], LR, EPOCHS)

    evaluate(model, dataset['test_matched'])
