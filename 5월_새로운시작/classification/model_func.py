from transformers import get_scheduler
from transformers import RobertaForSequenceClassification
from datasets import load_metric
from tqdm import tqdm
import torch
from transformers import AdamW
import os
def model_train(model_name, train_dataloader, dev_dataloader, device,save_path, num_epochs, num_label) :
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
    try:
        os.mkdir(save_path)
    except:
        pass
    best_model_path = "BestModel.pth"
    final_path = save_path + best_model_path
    for epoch in range(num_epochs):
        print(f"epoch : {epoch}")
        metric = load_metric("accuracy")
        losss = 0
        loss_count = 0
        for batch in tqdm(train_dataloader,desc="모델 학습중 "):
            model.train()
            optimizer.zero_grad()
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)

            loss = outputs.loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            losss += loss.item()
            loss_count +=1
        # 성능확인

        with torch.no_grad():
            for batch in tqdm(dev_dataloader, desc="모델 검증 중"):
                model.eval()
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                logits = outputs.logits
                predict = torch.argmax(logits, dim=-1)
                metric.add_batch(predictions=predict, references=batch["labels"])

        history = metric.compute()
        acc = history['accuracy']
        print(f"validation Accuracy => {acc}")
        print(f"Loss => {losss/loss_count}")
        accuracy_list.append(acc)

        if max(accuracy_list) <= acc :
            torch.save(model, final_path)


    return best_model_path, accuracy_list

