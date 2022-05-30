from transformers import (
    T5Tokenizer,
)
from tqdm import tqdm

def beam(model, device, input_ids, input_mask,setting_length) :
    beam_outputs = model.generate(
        input_ids=input_ids.to(device), attention_mask=input_mask.to(device),
        early_stopping=True,
        num_beams=10, #  Beam Search 각 타임스텝에서 가장 가능성 있는 num_beams개의 시퀀스를 유지하고, 최종적으로 가장 확률이 높은 가설을 선택하는 방법
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        max_length = setting_length
    )
    return beam_outputs

def nucleus(model, device, input_ids, input_mask, setting_length):
    nucleus_outputs = model.generate(
        input_ids=input_ids.to(device), attention_mask=input_mask.to(device),
        do_sample=True,  # 샘플링 전략 사용
        max_length=setting_length,  # 최대 디코딩 길이는 50
        top_k=50,  # 확률 순위가 50위 밖인 토큰은 샘플링에서 제외
        top_p=0.95,  # 누적 확률이 95%인 후보집합에서만 생성
        num_return_sequences=1,  # n개의 결과를 디코딩해낸다
        early_stopping=True
    )
    return nucleus_outputs

def generation_sentence(model, dataset, device, decoder_argorithm,setting_length, model_name = 't5-base') :
    tokenizer = T5Tokenizer.from_pretrained(model_name)

    g_model = model
    g_model.to(device)
    raw_dataset = dataset

    g_model.eval()

    repository = []

    for data in tqdm(raw_dataset,desc='데이터 증대 진행률 : ') :
        test_sent = 'falsify: {} </s>'.format(data[0])

        test_tokenized = tokenizer.encode_plus(test_sent, return_tensors="pt")

        test_input_ids = test_tokenized["input_ids"]
        test_attention_mask = test_tokenized["attention_mask"]

        if decoder_argorithm == "beam_search" :
            outputs = beam(g_model, device, test_input_ids, test_attention_mask,setting_length)
        elif decoder_argorithm == "nucleus_sampling":
            outputs = nucleus(g_model, device, test_input_ids, test_attention_mask,setting_length)

        for beam_output in outputs:
            sent = tokenizer.decode(beam_output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            repository.append([data[0], sent])  # 원래 문장, 생성된 문장

    return repository