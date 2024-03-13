from utils.load_dataset import load_data_ReadMe, make_dataloader_ReadMe
from utils.evaluation import evaluate
from utils.metrics import f1_score_func, metrics_per_class
from transformers import BertForSequenceClassification
import torch
import os

model = BertForSequenceClassification.from_pretrained(
    "bert-base-multilingual-cased",
    num_labels=6,
    output_attentions=False,
    output_hidden_states=True,
)

if not os.path.exists("./ReadMe_en"):
    print("Trained weights not found. Downloading once from net.")
    os.system(
        """wget "https://getfile.dokpub.com/yandex/get/https://disk.yandex.com/d/UN5pmzv66OM1Pg" -O ./ReadMe_en"""
    )


stete_dict = torch.load("./ReadMe_en", map_location="cpu")
model.load_state_dict(stete_dict)


device = "cpu"
if torch.cuda.is_available():
    device = "cuda"

model = model.to(device)

data = load_data_ReadMe("ru")
label_dict = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5}
dataloader_predict = make_dataloader_ReadMe(data, 10)

predict_loss, predictions, true_predict = evaluate(dataloader_predict)
predict_f1 = f1_score_func(predictions, true_predict)
print(f"F1 Score (Weighted): {predict_f1}")
metrics_per_class(predictions, true_predict, label_dict)
