import torch
from torch.utils.data import DataLoader
from torch import nn
from typing import List

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"


def get_neurons_activations(model: nn.Module, dataloader: DataLoader) -> torch.Tensor:
    model.eval()

    # Подготовка списка для хранения усредненных скрытых состояний по слоям
    avg_hidens_per_layer = [
        [] for _ in range(13)
    ]  # предполагается, что у модели 12 слоев + входной слой

    for batch in dataloader:
        batch = tuple(b.to(device) for b in batch)
        inputs = {
            "input_ids": batch[0],
            "attention_mask": batch[1],
            "labels": batch[2],
        }

        with torch.no_grad():
            outputs = model(**inputs)

        # Создание маски для исключения [PAD] токенов (где input_ids равно 0)
        mask = (
            inputs["input_ids"] != 0
        )  # Создается маска размером как input_ids, True где не PAD
        mask = mask.unsqueeze(-1).expand_as(
            outputs["hidden_states"][0]
        )  # Расширение маски до размера скрытых состояний

        for layer_idx, layer_hidden_states in enumerate(outputs["hidden_states"]):
            # Применение маски к скрытым состояниям
            masked_hidden_states = layer_hidden_states * mask.float()
            # Вычисление суммы и количества не-pad токенов для усреднения
            sum_hidden_states = masked_hidden_states.sum(dim=1)  # Сумма по оси токенов
            non_pad_tokens = mask.sum(dim=1)  # Количество не-pad токенов
            # Усреднение скрытых состояний, исключая pad-токены
            avg_hidden_states = sum_hidden_states / non_pad_tokens.clamp(
                min=1
            )  # Избегание деления на 0
            avg_hidens_per_layer[layer_idx].append(avg_hidden_states)

    # Собираем усредненные скрытые состояния по всему датасету для каждого слоя
    avg_hidens_stacked_per_layer = [
        torch.cat(layer_avg_hidens, dim=0) for layer_avg_hidens in avg_hidens_per_layer
    ]

    # Стекинг усредненных скрытых состояний для всех слоев
    all_avg_hidens_tensor = torch.stack(avg_hidens_stacked_per_layer)

    return all_avg_hidens_tensor  # [layers, batch, neurons]


def pearson_corr(x: torch.Tensor, y: torch.Tensor) -> float:
    return (torch.dot(x, y) / (torch.norm(x) * torch.norm(y))).item()


def ANC(L1: torch.Tensor, L2: torch.Tensor) -> float:
    X = L1 - L1.mean(dim=0, keepdim=True)
    Y = L2 - L2.mean(dim=0, keepdim=True)

    anc = 0
    for i in range(X.size()[0]):
        anc += abs(pearson_corr(X[i], Y[i]))

    return anc / X.size()[0]


def count_ANC(layers1: torch.Tensor, layers2: torch.Tensor) -> List[float]:
    anc = []
    for i in range(13):
        anc.append(ANC(layers1[i], layers2[i]))

    return anc
