from datasets import load_dataset

class snli_data():
    def __init__(self):
        self.true_false_adjective_tuples0 = []
        self.true_false_adjective_tuples1 = []
        self.true_false_adjective_tuples2 = []
        self.raw_datsets = load_dataset("snli")
        self.entailment = []  # 0
        self.neutral = []  # 1
        self.contradiction = []  # 2
        if self.raw_datsets:
            self.split_data()

    def split_data(self):
        self.raw_datasets['train'] = self.raw_datasets['train'].filter(lambda x: x['label'] in [1, 2, 0])

        for data in self.raw_datasets['train']:
            if data['label']==0 :
                self.entailment.append([data['premise'], data['hypothesis']])
            if data['label'] == 1:
                self.neutral.append([data['premise'], data['hypothesis']])
            if data['label'] == 2:
                self.contradiction.append([data['premise'], data['hypothesis']])

        # few-shot learning 하기위한 예제 입력
        for i in range(len(self.entailment)):
            self.true_false_adjective_tuples0.append((self.entailment[i][0], self.entailment[i][1]))
        for i in range(len(self.neutral)):
            self.true_false_adjective_tuples1.append((self.neutral[i][0], self.neutral[i][1]))
        for i in range(len(self.contradiction)):
            self.true_false_adjective_tuples2.append((self.contradiction[i][0], self.contradiction[i][1]))

        #return self.true_false_adjective_tuples0, self.true_false_adjective_tuples1, self.true_false_adjective_tuples2

    def load_entailment(self):
        return self.true_false_adjective_tuples0
    def load_neutral(self):
        return self.true_false_adjective_tuples1
    def load_contradiction(self):
        return self.true_false_adjective_tuples2