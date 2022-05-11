from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
from transformers import get_scheduler
import transformers
import random
import numpy as np

model_name = 'sentence-transformers/stsb-roberta-large'

from transformers import RobertaForSequenceClassification
model =RobertaForSequenceClassification.from_pretrained(model_name, num_labels=3)


print(model.state_dict())
