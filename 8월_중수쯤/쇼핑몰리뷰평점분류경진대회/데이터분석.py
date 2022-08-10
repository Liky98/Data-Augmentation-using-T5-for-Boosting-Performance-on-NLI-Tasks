import pandas as pd
from konlpy.tag import Okt
from tokenizers import SentencePieceBPETokenizer
import os
from transformers import RobertaConfig, DataCollatorWithPadding, RobertaForSequenceClassification, RobertaTokenizer
from datasets import load_dataset, DatasetDict

class tokenizer_class() :
    def csv_to_text(self) :
        data = pd.read_csv("dataset/train.csv")
        train_pair = [(row[1], row[2]) for _, row in data.iterrows() if type(row[1]) == str]  # nan 제거

        # 문장 및 라벨 데이터 추출
        train_data  = [pair[0] for pair in train_pair]
        train_label = [pair[1] for pair in train_pair]

        print('문장: %s' %(train_data[:3]))
        print('라벨: %s' %(train_label[:3]))

        # subword 학습을 위해 문장만 따로 저장
        with open('train_tokenizer.txt', 'w', encoding='utf-8') as f:
            for line in train_data:
                f.write(line+'\n')

        # subword 학습을 위해 문장만 따로 저장
        with open('train_tokenizer.txt', 'r', encoding='utf-8') as f:
            test_tokenizer = f.read().split('\n')
        print(test_tokenizer[:3])

        num_word_list = [len(sentence.split()) for sentence in test_tokenizer]
        print('\n코퍼스 평균/총 단어 갯수 : %.1f / %d' % (sum(num_word_list)/len(num_word_list), sum(num_word_list)))
        return test_tokenizer

    def Mecab(self, data):
        tokenizer = Okt()
        print('Okt check :', tokenizer.morphs('어릴때보고 지금다시봐도 재밌어요ㅋㅋ'))

        for_generation = False  # or normal

        if for_generation:
            # 1: '어릴때' -> '어릴, ##때' for generation model
            total_morph = []
            for sentence in data:
                # 문장단위 mecab 적용
                morph_sentence = []
                count = 0
                for token_mecab in tokenizer.morphs(sentence):
                    token_mecab_save = token_mecab
                    if count > 0:
                        token_mecab_save = "##" + token_mecab_save  # 앞에 ##를 부친다
                        morph_sentence.append(token_mecab_save)
                    else:
                        morph_sentence.append(token_mecab_save)
                        count += 1
                # 문장단위 저장
                total_morph.append(morph_sentence)

        else:
            # 2: '어릴때' -> '어릴, 때'   for normal case
            total_morph = []
            for sentence in data:
                # 문장단위 mecab 적용
                morph_sentence = tokenizer.morphs(sentence)
                # 문장단위 저장
                total_morph.append(morph_sentence)

        print(total_morph[:3])
        print(len(total_morph))

        # mecab 적용한 데이터 저장
        # ex) 1 line: '어릴 때 보 고 지금 다시 봐도 재밌 어요 ㅋㅋ'
        with open('after_mecab.txt', 'w', encoding='utf-8') as f:
            for line in total_morph:
                f.write(' '.join(line) + '\n')
        with open('after_mecab.txt', 'r', encoding='utf-8') as f:
            test_tokenizer = f.read().split('\n')

        return test_tokenizer

    # def add_special_tokens(self) :
    #     user_defined_symbols = ['[BOS]', '[EOS]', '[UNK0]',
    #                             '[UNK1]', '[UNK2]', '[UNK3]',
    #                             '[UNK4]', '[UNK5]', '[UNK6]',
    #                             '[UNK7]', '[UNK8]', '[UNK9]']
    #     unused_token_num = 200
    #     unused_list = ['[unused{}]'.format(n) for n in range(unused_token_num)]
    #     user_defined_symbols = user_defined_symbols + unused_list
    #
    #     print(user_defined_symbols)
    #     return user_defined_symbols

    def train_tokenizer(self, tokenizer) :
        corpus_file = ['after_mecab.txt']  # data path
        vocab_size = 32000
        limit_alphabet = 6000 #merge 수행 전 initial tokens이 유지되는 숫자 제한
        # output_path = 'hugging_%d' % (vocab_size)
        min_frequency = 2 # n회 이상 등장한 pair만 merge 수행

        # Then train it!
        tokenizer.train(files=corpus_file,
                        vocab_size=vocab_size,
                        min_frequency=min_frequency,  # 단어의 최소 발생 빈도, 5
                        limit_alphabet=limit_alphabet,  # ByteLevelBPETokenizer 학습시엔 주석처리 필요
                        show_progress=True,
                        special_tokens=["<s>", "</s>", "<unk>", "<pad>", "<mask>"])
        print('train complete')

        sentence = '나는 오늘 아침밥을 먹었다.'
        output = tokenizer.encode(sentence)
        print(sentence)
        print(output)

        # save tokenizer
        hf_model_path = 'liky_tokenizer'
        if not os.path.isdir(hf_model_path):
            os.mkdir(hf_model_path)

        tokenizer.save_model(hf_model_path)  # vocab.txt 파일 한개가 만들어진다

        return tokenizer

    def test(self):
        from transformers import RobertaTokenizer

        tokenizer_for_load = RobertaTokenizer.from_pretrained('liky_tokenizer',
                                                               strip_accents=False,  # Must be False if cased model
                                                               lowercase=False)  # 로드

        print('vocab size : %d' % tokenizer_for_load.vocab_size)
        # tokenized_input_for_pytorch = tokenizer_for_load("i am very hungry", return_tensors="pt")
        tokenized_input_for_pytorch = tokenizer_for_load("나는 오늘 아침밥을 먹었다.", return_tensors="pt")
        tokenized_input_for_tensorflow = tokenizer_for_load("나는 오늘 아침밥을 먹었다.", return_tensors="tf")

        print("Tokens (str)      : {}".format([tokenizer_for_load.convert_ids_to_tokens(s) for s in
                                               tokenized_input_for_pytorch['input_ids'].tolist()[0]]))
        print("Tokens (int)      : {}".format(tokenized_input_for_pytorch['input_ids'].tolist()[0]))
        print("Tokens (attn_mask): {}\n".format(tokenized_input_for_pytorch['attention_mask'].tolist()[0]))


if __name__ =="__main__":
    tokenizer_class = tokenizer_class()

    text_data = tokenizer_class.csv_to_text()
    text_data = tokenizer_class.Mecab(text_data)

    tokenizer = SentencePieceBPETokenizer()
    tokenizer = tokenizer_class.train_tokenizer(tokenizer)

    config = RobertaConfig(
        vocab_size=22745,
        max_position_embeddings=514,
        num_attention_heads=12,
        num_hidden_layers=6,
        type_vocab_size=1,
    )

    model = RobertaForSequenceClassification(config=config)
    print('Num parameters: ', model.num_parameters())
    tokenizer = RobertaTokenizer.from_pretrained("./liky_tokenizer")
    print(tokenizer)

    raw_train = load_dataset('csv', data_files='./dataset/train.csv')
    raw_test = load_dataset('csv', data_files='./dataset/test.csv')
    train, valid = raw_train['train'].train_test_split(test_size=0.1).values()
    dataset = DatasetDict({'train': train, 'valid': valid, 'test': raw_test['train']})
    print(dataset)

    print(tokenizer.tokenize(train['reviews'][0]))


    def tokenize_function(example):
        return tokenizer(example["reviews"], truncation=True)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    print(tokenized_datasets)
    tokenized_datasets = tokenized_datasets.remove_columns(["id", "reviews"])
    tokenized_datasets['train'] = tokenized_datasets['train'].rename_column("target", "labels")
    tokenized_datasets['valid'] = tokenized_datasets['valid'].rename_column("target", "labels")
    tokenized_datasets.set_format("torch")
    tokenized_datasets["train"].column_names

    from torch.utils.data import DataLoader

    train_dataloader = DataLoader(tokenized_datasets["train"], shuffle=True, batch_size=4,
                                  collate_fn=data_collator)
    valid_dataloader = DataLoader(tokenized_datasets["valid"], shuffle=True, batch_size=4,
                                  collate_fn=data_collator)
    eval_dataloader = DataLoader(tokenized_datasets["test"], shuffle=False, batch_size=4,
                                 collate_fn=data_collator)

    for batch in train_dataloader:
        break
    {k: v.shape for k, v in batch.items()}
#%%
# tokenizer = RobertaTokenizer("./liky_tokenizer/vocab.json", "./liky_tokenizer/merges.txt")
#
# print(tokenizer.encode(train['reviews'][0]))
# train['reviews'][0]
# #tokenizer
token_class = tokenizer_class
data = token_class.csv_to_text()
token = token_class.Mecab(data)
token_class.train_tokenizer(token)
