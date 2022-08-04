from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch.optim import Adam
import SNLI_load_and_processing
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
import torch

def processing():
    """ 데이터 전처리 """
    raw_datasets = load_dataset("snli")

    raw_datasets['train'] = raw_datasets['train'].filter(lambda x : x['label'] in [1, 2, 0])
    raw_datasets['validation'] = raw_datasets['validation'].filter(lambda x : x['label'] in [1, 2, 0])
    raw_datasets['test'] = raw_datasets['test'].filter(lambda x : x['label'] in [1, 2, 0])

    entailment =[] #0
    neutral =[] #1
    contradiction = [] #2
    for data in tqdm(raw_datasets['train']) :
        if data['label'] == 0 :
            entailment.append([data['premise'], data['hypothesis']])
        if data['label'] == 1:
            neutral.append([data['premise'], data['hypothesis']])
        if data['label'] == 2:
            contradiction.append([data['premise'], data['hypothesis']])

    random.shuffle(entailment)
    random.shuffle(neutral)
    random.shuffle(contradiction)

    return entailment, neutral, contradiction


def few_shot_dataset(dataset, label):
    few_shot_train_sample = []
    few_shot_val_sample = []

    for i in tqdm(range(len(dataset))):
        if len(few_shot_train_sample) <100 :
            few_shot_train_sample.append([label+dataset[i][0], dataset[i][1]])

        elif len(few_shot_val_sample) < 30 :
            few_shot_val_sample.append([label+dataset[i][0], dataset[i][1]])

        else : break

    return few_shot_train_sample, few_shot_val_sample

def result_graph(train_loss_list, val_loss_list):
    epochs = [x for x in range(len(train_loss_list))]
    plt.plot(epochs, train_loss_list, 'r', label='Training loss')
    epochs = [x for x in range(len(val_loss_list))]
    plt.plot(epochs, val_loss_list, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

if __name__ =="__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    try :
        print(f"데이터셋확인 \n{Entailment_train_sample}")
    except :
        entailment, neutral, contradiction = processing()
        Entailment_train_sample, Entailment_val_sample =few_shot_dataset(entailment, "Entailment: ")
        Neutral_train_sample, Neutral_val_sample = few_shot_dataset(neutral, "Neutral: ")
        Contradiction_train_sample, Contradiction_val_sample = few_shot_dataset(contradiction, "Contradiction: ")

    model = T5ForConditionalGeneration.from_pretrained("t5-base").to(device)
    tokenizer = T5Tokenizer.from_pretrained('t5-base')
    optimizer = Adam(model.parameters(), lr = 3e-4)

    train_loss = []
    val_loss = []

    for i in range(1):
        model.train()
        for input_sentence, output_sentence in tqdm(Contradiction_train_sample) :
            inputs = tokenizer(input_sentence, return_tensors="pt").to(device)
            outputs = tokenizer(output_sentence, return_tensors='pt').to(device)

            optimizer.zero_grad()
            output = model(input_ids=inputs.input_ids,
                           attention_mask=inputs.attention_mask,
                           labels=outputs.input_ids,
                           decoder_attention_mask= outputs.attention_mask)

            output.loss.backward()
            optimizer.step()

            predict = torch.argmax(output.logits, dim=-1)
            train_loss.append(output.loss.item())

        with torch.no_grad():
            model.eval()
            for input_sentence, output_sentence in tqdm(Contradiction_val_sample) :
                inputs = tokenizer(input_sentence, return_tensors="pt").to(device)
                outputs = tokenizer(output_sentence, return_tensors='pt').to(device)

                # output = model.generate(**inputs)

                output = model(input_ids=inputs.input_ids,
                               attention_mask=inputs.attention_mask,
                               labels=outputs.input_ids,
                               decoder_attention_mask=outputs.attention_mask)


                predict = torch.argmax(output.logits, dim=-1)
                val_loss.append(output.loss.item())
                print()
                print(input_sentence)
                print(output_sentence)
                print(tokenizer.decode(predict[0], skip_special_tokens=True))
    result_graph(train_loss, val_loss)


#%%
torch.save(model, "T5_Contradiction.pth")