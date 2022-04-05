from transformers import AutoConfig
from transformers import PretrainedConfig

config = PretrainedConfig.from_pretrained('roberta-base')
config

#config = AutoConfig.from_pretrained("bert-base-cased")