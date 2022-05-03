from transformers import get_scheduler
from transformers import RobertaForSequenceClassification
from datasets import load_metric
from tqdm import tqdm
import torch
from transformers import AdamW


def model_train(model_name, train_dataloader, dev_dataloader, device, save_path, num_label = 3, num_epochs = 3) :
    model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=num_label)

    # 옵티마이저 설정 및 스케줄러
    optimizer = AdamW(model.parameters(), lr=2e-5)

    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    for epoch in tqdm(range(num_epochs), desc="모델 학습중"):
        metric = load_metric("accuracy")

        for batch in train_dataloader:
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
        print(metric.compute())

    torch.save(model, save_path)

    return model
