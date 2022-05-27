import torch
from torch import nn
from transformers.modeling_roberta import RobertaModel, RobertaConfig

from datasets.collate_functions import collate_to_max_length


class ExplainableModel(nn.Module):
    def __init__(self, bert_dir):
        super().__init__()
        self.bert_config = RobertaConfig.from_pretrained(bert_dir, output_hidden_states=False)
        self.intermediate = RobertaModel.from_pretrained(bert_dir)
        self.span_info_collect = SICModel(self.bert_config.hidden_size)
        self.interpretation = InterpretationModel(self.bert_config.hidden_size)
        self.output = nn.Linear(self.bert_config.hidden_size, self.bert_config.num_labels)

    def forward(self, input_ids, start_indexs, end_indexs, span_masks):
        # generate mask
        attention_mask = (input_ids != 1).long()
        # intermediate layer
        hidden_states, first_token = self.intermediate(input_ids, attention_mask=attention_mask)  # output.shape = (bs, length, hidden_size)
        # span info collecting layer(SIC)
        h_ij = self.span_info_collect(hidden_states, start_indexs, end_indexs)
        # interpretation layer
        H, a_ij = self.interpretation(h_ij, span_masks)
        # output layer
        out = self.output(H)
        return out, a_ij

def main():
    # data
    input_id_1 = torch.LongTensor([0, 4, 5, 6, 7, 2])
    input_id_2 = torch.LongTensor([0, 4, 5, 2])
    input_id_3 = torch.LongTensor([0, 4, 2])
    batch = [(input_id_1, torch.LongTensor([1]), torch.LongTensor([6])),
             (input_id_2, torch.LongTensor([1]), torch.LongTensor([4])),
             (input_id_3, torch.LongTensor([1]), torch.LongTensor([3]))]

    output = collate_to_max_length(batch=batch, fill_values=[1, 0, 0])
    input_ids, labels, length, start_indexs, end_indexs, span_masks = output

    # model
    bert_path = "/data/nfsdata2/sunzijun/loop/roberta-base"
    model = ExplainableModel(bert_path)
    print(model)

    output = model(input_ids, start_indexs, end_indexs, span_masks)
    print(output)


if __name__ == '__main__':
    main()