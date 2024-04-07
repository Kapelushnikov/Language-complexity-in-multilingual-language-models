import hydra
import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from pytorch_lightning.loggers import MLFlowLogger
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from transformers import BertForSequenceClassification, get_linear_schedule_with_warmup
from utils.load_dataset import load_data_ReadMe, make_dataloader_ReadMe

# Инициализация MLflow Logger
mlflow_logger = MLFlowLogger(
    experiment_name="ReadMeClassification", tracking_uri="http://localhost:5000"
)


class ReadMeDataModule(pl.LightningDataModule):
    def __init__(self, train_data, val_data, batch_size):
        super().__init__()
        self.train_data = train_data
        self.val_data = val_data
        self.batch_size = batch_size

    def train_dataloader(self):
        return make_dataloader_ReadMe(self.train_data, batch_size=self.batch_size)

    def val_dataloader(self):
        return make_dataloader_ReadMe(self.val_data, batch_size=self.batch_size)


class ReadMeModel(pl.LightningModule):
    def __init__(self, n_labels, learning_rate, eps):
        super().__init__()
        self.save_hyperparameters()
        self.model = BertForSequenceClassification.from_pretrained(
            "bert-base-multilingual-cased",
            num_labels=n_labels,
            output_attentions=False,
            output_hidden_states=True,
        )

    def forward(self, input_ids, attention_mask, labels=None):
        output = self.model(input_ids, attention_mask=attention_mask, labels=labels)
        return output

    def training_step(self, batch, batch_idx):
        inputs = {
            "input_ids": batch[0],
            "attention_mask": batch[1],
            "labels": batch[2],
        }
        outputs = self.forward(**inputs)
        loss = outputs[0]
        self.log("train_loss", loss)  # Логгирование потерь обучения
        return loss

    def validation_step(self, batch, batch_idx):
        inputs = {
            "input_ids": batch[0],
            "attention_mask": batch[1],
            "labels": batch[2],
        }
        outputs = self.forward(**inputs)
        val_loss, logits = outputs[:2]
        preds = torch.argmax(logits, dim=1)
        labels = inputs["labels"]
        self.log("val_loss", val_loss)  # Логгирование потерь валидации
        return {"val_loss": val_loss, "preds": preds, "labels": labels}


@hydra.main(config_path="config", config_name="config")
def main(cfg: DictConfig):
    # Подготовка данных
    data = load_data_ReadMe()
    data_train, data_val = train_test_split(data, test_size=0.2)
    n_labels = len(np.unique(data_train["Rating"]))

    datamodule = ReadMeDataModule(data_train, data_val, cfg.data.batch_size)

    # Инициализация модели
    model = ReadMeModel(
        n_labels=n_labels,
        learning_rate=cfg.training.learning_rate,
        eps=cfg.training.eps,
    )

    # Обучение
    trainer = pl.Trainer(
        logger=mlflow_logger,
        max_epochs=cfg.training.epochs,
        gpus=1 if torch.cuda.is_available() else 0,
    )
    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
