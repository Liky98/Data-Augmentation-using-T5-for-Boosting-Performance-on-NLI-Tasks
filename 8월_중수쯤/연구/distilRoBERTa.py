from transformers import DistilBertTokenizer, DistilBertModel
import torch

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertModel.from_pretrained("distilbert-base-uncased")

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

outputs,x = model(**inputs)

print(outputs)
print(x)
last_hidden_states = outputs.last_hidden_state
#%%
print(inputs)
print(inputs.input_ids[0])
print(tokenizer.decode(inputs.input_ids[0]))
print(last_hidden_states.shape)
#%%
x = torch.randn(3,4)
#%%
print(f' x > {x}')
print(f'x_shape > {x.shape}')
indices = torch.tensor([1,2])
print(f'Indices > {indices}')

print(f'index_select > {torch.index_select(x, 0, indices)}')
#%%
from typing import List

import numpy as np
import torch


def collate_to_max_length(batch: List[List[torch.Tensor]], max_len: int = None, fill_values: List[float] = None) -> \
    List[torch.Tensor]:
    """
    pad to maximum length of this batch
    Args:
        batch: a batch of samples, each contains a list of field data(Tensor), which shape is [seq_length]
        max_len: specify max length
        fill_values: specify filled values of each field
    Returns:
        output: list of field batched data, which shape is [batch, max_length]
    """
    # [batch, num_fields]
    lengths = np.array([[len(field_data) for field_data in sample] for sample in batch])
    batch_size, num_fields = lengths.shape
    fill_values = fill_values or [0.0] * num_fields
    # [num_fields]
    max_lengths = lengths.max(axis=0)
    if max_len:
        assert max_lengths.max() <= max_len
        max_lengths = np.ones_like(max_lengths) * max_len

    output = [torch.full([batch_size, max_lengths[field_idx]],
                         fill_value=fill_values[field_idx],
                         dtype=batch[0][field_idx].dtype)
              for field_idx in range(num_fields)]
    for sample_idx in range(batch_size):
        for field_idx in range(num_fields):
            # seq_length
            data = batch[sample_idx][field_idx]
            output[field_idx][sample_idx][: data.shape[0]] = data
    # generate span_index and span_mask
    max_sentence_length = max_lengths[0]
    start_indexs = []
    end_indexs = []
    for i in range(1, max_sentence_length - 1):
        for j in range(i, max_sentence_length - 1):
            # # span大小为10
            # if j - i > 10:
            #     continue
            start_indexs.append(i)
            end_indexs.append(j)
    # generate span mask
    span_masks = []
    for input_ids, label, length in batch:
        span_mask = []
        middle_index = input_ids.tolist().index(2)
        for start_index, end_index in zip(start_indexs, end_indexs):
            if 1 <= start_index <= length.item() - 2 and 1 <= end_index <= length.item() - 2 and (
                start_index > middle_index or end_index < middle_index):
                span_mask.append(0)
            else:
                span_mask.append(1e6)
        span_masks.append(span_mask)
    # add to output
    output.append(torch.LongTensor(start_indexs))
    output.append(torch.LongTensor(end_indexs))
    output.append(torch.LongTensor(span_masks))
    return output  # (input_ids, labels, length, start_indexs, end_indexs, span_masks)


if __name__ == '__main__':
    input_id_1 = torch.LongTensor([0, 3, 2, 5, 6, 2])
    input_id_2 = torch.LongTensor([0, 3, 2, 4, 2])
    input_id_3 = torch.LongTensor([0, 3, 2])
    batch = [(input_id_1, torch.LongTensor([1]), torch.LongTensor([6])),
             (input_id_2, torch.LongTensor([1]), torch.LongTensor([5])),
             (input_id_3, torch.LongTensor([1]), torch.LongTensor([4]))]
    output = collate_to_max_length(batch=batch, fill_values=[1, 99, 0])
    a,b,c,d,e,f = output
    print(a)
    print(b)
    print(c)
    print(d)
    print(e)
    print(f)
    #%%
    x = torch.randn([1,8,20])
    print(x)
    print(x.pow(2))
    print(x.pow(2).sum(dim=-1))
    x.pow(2).sum(dim=-1).mean()

    #%%
    from transformers import BertTokenizer, BertModel
    import torch

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")

    inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

    outputs, x = model(**inputs)
    output = model(**inputs)

    print(outputs)
    print(x)
    print(output.pooler_output.shape)