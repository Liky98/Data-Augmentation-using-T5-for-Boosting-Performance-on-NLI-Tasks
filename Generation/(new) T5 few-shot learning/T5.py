"""
Prepare Model
"""

import random
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5Tokenizer,
    get_linear_schedule_with_warmup
)

pl.seed_everything(42)

t5_tokenizer = T5Tokenizer.from_pretrained('t5-base')
t5_model = T5ForConditionalGeneration.from_pretrained('t5-base')

# dataset preparation

true_false_adjective_tuples_train = [
    ("The cat is alive", "The cat is dead"),
    ("The old woman is beautiful", "The old woman is ugly"),
    ("The purse is cheap", "The purse is expensive"),
    ("Her hair is curly", "Her hair is straight"),
    ("The bathroom is clean", "The bathroom is dirty"),
    ("The exam was easy", "The exam was difficult"),
    ("The house is big", "The house is small"),
    ("The house owner is good", "The house owner is bad"),
    ("The little kid is fat", "The little kid is thin"),
    ("She arrived early", "She arrived late."),
    ("John is very hardworking", "John is very lazy"),
    ("The fridge is empty", "The fridge is full")

]

true_false_adjective_tuples_validation = [
    ("Her face was bright", "Her face was dull"),
    ("The kid is very active", "The kid is very silent")

]

# 데이터셋
from tqdm.notebook import tqdm
import copy


class FalseGenerationDataset(Dataset):
    def __init__(self, tokenizer, tf_list, max_len_inp=96, max_len_out=96):
        self.true_false_adjective_tuples = tf_list

        self.max_len_input = max_len_inp
        self.max_len_output = max_len_out
        self.tokenizer = tokenizer
        self.inputs = []
        self.targets = []
        self.skippedcount = 0
        self._build()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        source_ids = self.inputs[index]["input_ids"].squeeze()
        target_ids = self.targets[index]["input_ids"].squeeze()

        src_mask = self.inputs[index]["attention_mask"].squeeze()  # might need to squeeze
        target_mask = self.targets[index]["attention_mask"].squeeze()  # might need to squeeze

        labels = copy.deepcopy(target_ids)
        labels[labels == 0] = -100

        return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids, "target_mask": target_mask,
                "labels": labels}

    def _build(self):
        for inputs, outputs in self.true_false_adjective_tuples:
            input_sent = "falsify: " + inputs
            ouput_sent = "falsified: " + outputs

            # tokenize inputs
            tokenized_inputs = self.tokenizer.batch_encode_plus(
                [input_sent], max_length=self.max_len_input, pad_to_max_length=True, return_tensors="pt"
            )
            # tokenize targets
            tokenized_targets = self.tokenizer.batch_encode_plus(
                [ouput_sent], max_length=self.max_len_output, pad_to_max_length=True, return_tensors="pt"
            )

            self.inputs.append(tokenized_inputs)
            self.targets.append(tokenized_targets)


train_dataset = FalseGenerationDataset(t5_tokenizer,true_false_adjective_tuples_train)
validation_dataset = FalseGenerationDataset(t5_tokenizer,true_false_adjective_tuples_validation)

# 모델 파인튜닝
class T5FineTuner(pl.LightningModule):
    def __init__(self, hparams, t5model, t5tokenizer):
        super(T5FineTuner, self).__init__()
        self.hparams = hparams
        self.model = t5model
        self.tokenizer = t5tokenizer

    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None, decoder_attention_mask=None,
                lm_labels=None):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=decoder_attention_mask,
            labels=lm_labels,
        )

        return outputs

    def training_step(self, batch, batch_idx):
        outputs = self.forward(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            decoder_input_ids=batch["target_ids"],
            decoder_attention_mask=batch['target_mask'],
            lm_labels=batch['labels']
        )

        loss = outputs[0]
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self.forward(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            decoder_input_ids=batch["target_ids"],
            decoder_attention_mask=batch['target_mask'],
            lm_labels=batch['labels']
        )

        loss = outputs[0]
        self.log("val_loss", loss)
        return loss

    def train_dataloader(self):
        return DataLoader(train_dataset, batch_size=self.hparams.batch_size, num_workers=4)

    def val_dataloader(self):
        return DataLoader(validation_dataset, batch_size=self.hparams.batch_size, num_workers=4)

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=3e-4, eps=1e-8)
        return optimizer

# 모델 학습
import argparse
args_dict = dict(
    batch_size=1,
)

args = argparse.Namespace(**args_dict)


model = T5FineTuner(args,t5_model,t5_tokenizer)

trainer = pl.Trainer(max_epochs = 5, gpus=1,progress_bar_refresh_rate=30)

trainer.fit(model)

# 테스트
test_sent = 'falsify: The sailor was happy and joyful. </s>'
test_tokenized = t5_tokenizer.encode_plus(test_sent, return_tensors="pt")

test_input_ids  = test_tokenized["input_ids"]
test_attention_mask = test_tokenized["attention_mask"]

model.model.eval()
beam_outputs = model.model.generate(
    input_ids=test_input_ids,attention_mask=test_attention_mask,
    max_length=64,
    early_stopping=True,
    num_beams=10,
    num_return_sequences=3,
    no_repeat_ngram_size=2
)

for beam_output in beam_outputs:
    sent = t5_tokenizer.decode(beam_output, skip_special_tokens=True,clean_up_tokenization_spaces=True)
    print (sent)
