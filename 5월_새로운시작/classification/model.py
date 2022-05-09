from transformers import get_scheduler
from transformers import RobertaForSequenceClassification
from datasets import load_metric
from tqdm import tqdm
import torch
from transformers import AdamW
import numpy as np

def model_train(model_name, train_dataloader, dev_dataloader, device, save_path,  num_epochs = 3, num_label = 3) :
    model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=num_label)
    model.to(device)
    # 옵티마이저 설정 및 스케줄러
    optimizer = AdamW(model.parameters(), lr=2e-5)

    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    accuracy_list = []
    for epoch in range(num_epochs):
        history = []
        mean_history = []
        print(f"epoch : {epoch}")
        metric = load_metric("accuracy")

        for batch in tqdm(train_dataloader,desc="모델 학습중 "):
            model.train()
            optimizer.zero_grad()
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

        # 성능확인
        model.eval()
        with torch.no_grad():
            for batch in tqdm(dev_dataloader, desc="모델 검증 중"):
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                logits = outputs.logits
                predict = torch.argmax(logits, dim=-1)
                metric.add_batch(predictions=predict, references=batch["labels"])

        history.append(metric.compute())
        for data in history:
            mean_history.append(data['accuracy'])

        average = np.mean(mean_history)
        print(f"validation Accuracy => {average}")

        accuracy_list.append(average)
        modelpath = save_path+epoch
        torch.save(model, modelpath)

    return model, accuracy_list
