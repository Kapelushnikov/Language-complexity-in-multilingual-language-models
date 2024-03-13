from utils.load_dataset import load_data_ReadMe, make_dataloader_ReadMe
from sklearn.model_selection import train_test_split
from transformers import (
    BertForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup,
)

import numpy as np
import torch

data = load_data_ReadMe()
data_train, data_val = train_test_split(data, test_size=0.2)

n_labels = len(np.unique(data_train["Rating"]))
label_dict = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5}

dataloader_train = make_dataloader_ReadMe(data_train, 10)
dataloader_val = make_dataloader_ReadMe(data_val, 10)

model = BertForSequenceClassification.from_pretrained(
    "bert-base-multilingual-cased",
    num_labels=n_labels,
    output_attentions=False,
    output_hidden_states=True,
)


optimizer = AdamW(model.parameters(), lr=1e-5, eps=1e-8)

scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=0, num_training_steps=len(dataloader_train) * epochs
)

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"

model = model.to(device)

epochs = 5

for epoch in range(1, epochs + 1):
    model.train()

    loss_train_total = 0

    progress_bar = tqdm(
        dataloader_train, desc="Epoch {:1d}".format(epoch), leave=False, disable=False
    )
    for batch in progress_bar:
        model.zero_grad()

        batch = tuple(b.to(device) for b in batch)

        inputs = {
            "input_ids": batch[0],
            "attention_mask": batch[1],
            "labels": batch[2],
        }

        outputs = model(**inputs)

        loss = outputs[0]
        loss_train_total += loss.item()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()

        progress_bar.set_postfix(
            {"training_loss": "{:.3f}".format(loss.item() / len(batch))}
        )

    tqdm.write(f"\nEpoch {epoch}")

    loss_train_avg = loss_train_total / len(dataloader_train)
    tqdm.write(f"Training loss: {loss_train_avg}")

    val_loss, predictions, true_vals = evaluate(dataloader_validation)
    val_f1 = f1_score_func(predictions, true_vals)
    tqdm.write(f"Validation loss: {val_loss}")
    tqdm.write(f"F1 Score (Weighted): {val_f1}")
    # accuracy_per_class(predictions, true_vals)
    plot_confusion_matrix(predictions, true_vals, label_dict)
    metrics_per_class(predictions, true_vals, label_dict)
