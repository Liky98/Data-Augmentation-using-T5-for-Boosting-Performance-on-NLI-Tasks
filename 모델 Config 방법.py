"""
허깅페이스 모델 가져와서 Transfer Learning 하는 방법
"""
### 1번 학습 방법 ######
import torch.nn as nn
from transformers import AutoModel

class PosModel(nn.Module):
    def __init__(self):
        super(PosModel, self).__init__()

        self.base_model = AutoModel.from_pretrained(model_name) # 허깅페이스 모델 이름
        self.dropout = nn.Dropout(0.5)
        self.linear = nn.Linear(768, 3)  # output features from bert is 768 and 2 is ur number of labels

    def forward(self, input_ids, attn_mask):
        outputs = self.base_model(input_ids, attention_mask=attn_mask)
        # You write you new head here
        outputs = self.dropout(outputs[0])
        outputs = self.linear(outputs)

        return outputs


model = PosModel()
model.to('cuda')

### 2번 학습 방법 ########
from transformers import AutoConfig
from transformers import AutoModelForSequenceClassification
config_model = AutoConfig.from_pretrained(model_name) # 허깅페이스 모델 이름
config_model.num_labels = 3
model =AutoModelForSequenceClassification.from_config(config_model)