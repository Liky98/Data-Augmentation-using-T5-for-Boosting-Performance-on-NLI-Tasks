from transformers import RobertaModel
import torch
from torch import nn
import argparse


class ClassificationHead(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.dense = nn.Linear(args.hidden_dim, args.hidden_dim)
        classifier_dropout = (args.drop_out if args.drop_out is not None else 0.1)
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(args.hidden_dim, args.num_labels)

    def forward(self, features):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class SequenceClassification(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.num_labels = args.num_labels
        self.model = RobertaModel.from_pretrained('roberta-base')
        self.classifier = ClassificationHead(args)

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0] #last hidden state
        logits = self.classifier(sequence_output)
        return logits