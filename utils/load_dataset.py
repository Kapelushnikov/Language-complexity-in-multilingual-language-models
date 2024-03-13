import torch
from transformers import BertTokenizer
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader, SequentialSampler
import pandas as pd

tokenizer = BertTokenizer.from_pretrained(
    "bert-base-multilingual-cased", do_lower_case=True
)


def load_data_ReadMe(language="en") -> pd.DataFrame:
    data = pd.concat(
        [
            pd.read_excel(
                f"./data/readme/dataset/{language}/readme_{language}_train.xlsx"
            ),
            pd.read_excel(
                f"./data/readme/dataset/{language}/readme_{language}_val.xlsx"
            ),
            pd.read_excel(
                f"./data/readme/dataset/{language}/readme_{language}_test.xlsx"
            ),
        ],
        axis=0,
    )
    return data


def make_dataloader_ReadMe(
    data: pd.DataFrame, batch_size=3, n_samples_per_class=None
) -> DataLoader:
    label_dict = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5}

    data_lang = pd.DataFrame(columns=["text", "label"])
    data_lang["text"] = data["Sentence"]
    data_lang["label"] = data["Rating"].replace(label_dict)

    if n_samples_per_class is not None:
        samples_list = []
        for label in label_dict.values():
            samples = data_lang[data_lang["label"] == label].sample(
                n=n_samples_per_class, replace=False
            )
            samples_list.append(samples)

        data_lang = pd.concat(samples_list)

    encoded_lang = tokenizer.batch_encode_plus(
        data_lang["text"].values,
        add_special_tokens=True,
        return_attention_mask=True,
        padding="max_length",
        max_length=256,
        return_tensors="pt",
    )

    input_ids_predict = encoded_lang["input_ids"]
    attention_masks_predict = encoded_lang["attention_mask"]
    labels_predict = torch.tensor(data_lang["label"].values)

    dataset_lang = TensorDataset(
        input_ids_predict, attention_masks_predict, labels_predict
    )

    dataloader_lang = DataLoader(
        dataset_lang, sampler=SequentialSampler(dataset_lang), batch_size=batch_size
    )
    return dataloader_lang
