import torch
from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5Tokenizer,
)
from tqdm import tqdm


# 모델 학습 함수 작성
def model_train(device, dataset, epochs, path, model_name = 't5-base', set_max_length = 96) :
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    t5_model = T5ForConditionalGeneration.from_pretrained(model_name) # 업데이트마다 모델 새로
    t5_model.to(device)
    t5_model.train()

    """optimizer 설정 """
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in t5_model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
        {
            "params": [p for n, p in t5_model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=3e-4, eps=1e-8)

    true_false_adjective_tuples = [(data[0], data[1]) for data in dataset]

    for epoch in range(epochs):
        print(f"epoch => {epoch+1}")
        for input, output in tqdm(true_false_adjective_tuples, desc='모델 학습 중 ') :
            input_sent =  "falsify: "+input +" </s>"
            ouput_sent = output +" </s>"

            tokenized_inp = tokenizer.encode_plus(input_sent, max_length=set_max_length, pad_to_max_length=True,
                                                  return_tensors="pt")
            tokenized_output = tokenizer.encode_plus(ouput_sent, max_length=set_max_length, pad_to_max_length=True,
                                                     return_tensors="pt")

            input_ids = tokenized_inp["input_ids"]
            attention_mask = tokenized_inp["attention_mask"]

            labels = tokenized_output["input_ids"]
            decoder_attention_mask = tokenized_output["attention_mask"]

            # forward 함수 -> decoder_input_ids 생성
            output = t5_model(input_ids=input_ids.to(device), labels=labels.to(device),
                              decoder_attention_mask=decoder_attention_mask.to(device),
                              attention_mask=attention_mask.to(device))
            loss = output[0]

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    torch.save(t5_model, '{}.pth'.format(path))

    return t5_model