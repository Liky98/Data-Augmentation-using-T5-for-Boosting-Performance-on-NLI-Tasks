from datasets import load_dataset
from sentence_transformers import InputExample

# entailment (0), neutral (1), contradiction (2)
dataset = load_dataset("multi_nli")

#%%
datase
#%%
train_samples = [] #392,702
validation_matched = [] #9,815
validation_mismatched = [] #9,832
for sentence in dataset :
    if sentence['split']=='train' :
        train_samples.append(InputExample(texts=[sentence['premise'], sentence['hypothesis']], label=sentence['label']))
    elif sentence['split'] == 'validation_matched' :
        validation_matched.append(InputExample(texts=[sentence['premise'], sentence['hypothesis']], label=sentence['label']))
    else :
        validation_mismatched.append(InputExample(texts=[sentence['premise'], sentence['hypothesis']], label=sentence['label']))

print(f"train : {len(train_samples)}")
print(f"matched val : {len(validation_matched)}")
print(f"mismatched val : {len(validation_mismatched)}")