import json
import os
from sklearn.metrics import (
    roc_auc_score,
)
import numpy as np
from sklearn.preprocessing import LabelBinarizer
import pandas as pd
import torch


def data2gpu(batch, use_cuda):
    if use_cuda:
        batch_data = {
            "content": batch[0].cuda(non_blocking=True),
            "content_masks": batch[1].cuda(non_blocking=True),
            "comments": batch[2].cuda(non_blocking=True),
            "comments_masks": batch[3].cuda(non_blocking=True),
            "content_emotion": batch[4].cuda(non_blocking=True),
            "comments_emotion": batch[5].cuda(non_blocking=True),
            "emotion_gap": batch[6].cuda(non_blocking=True),
            "style_feature": batch[7].cuda(non_blocking=True),
            "label": batch[8].cuda(non_blocking=True),
            "category": batch[9].cuda(non_blocking=True),
        }
    else:
        batch_data = {
            "content": batch[0],
            "content_masks": batch[1],
            "comments": batch[2],
            "comments_masks": batch[3],
            "content_emotion": batch[4],
            "comments_emotion": batch[5],
            "emotion_gap": batch[6],
            "style_feature": batch[7],
            "label": batch[8],
            "category": batch[9],
        }
    return batch_data


import torch


def category_logs(log, category_dict, mode, label, label_pred, loss, loss_fn, category):
    if category is not None:
        for k, v in category_dict.items():
            mask = category == v
            if torch.sum(mask) > 0:
                default_log(
                    log,
                    mode + "_" + k,
                    label[mask],
                    label_pred[mask],
                    loss_fn(label_pred[mask], label[mask].float()),
                )

    default_log(log, mode, label, label_pred, loss)


def default_log(log, mode, label, label_pred, loss):
    log(
        f"{mode}_loss",
        loss,
        on_step=False,
        on_epoch=True,
        prog_bar=True,
        logger=True,
    )
    accuracy = torch.sum(torch.round(label_pred) == label.float()).float() / len(label)
    log(
        f"{mode}_acc",
        accuracy,
        on_step=False,
        on_epoch=True,
        prog_bar=True,
        logger=True,
    )

    true_positives = torch.sum(
        (torch.round(label_pred) == label.float()) & (label.float() == 1)
    ).float()
    false_positives = torch.sum(
        (torch.round(label_pred) != label.float()) & (label.float() == 0)
    ).float()
    false_negatives = torch.sum(
        (torch.round(label_pred) != label.float()) & (label.float() == 1)
    ).float()

    precision = true_positives / (true_positives + false_positives + 1e-8)
    recall = true_positives / (true_positives + false_negatives + 1e-8)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)

    log(
        f"{mode}_f1",
        f1_score,
        on_step=False,
        on_epoch=True,
        logger=True,
    )

    log(
        f"{mode}_recall",
        recall,
        on_step=False,
        on_epoch=True,
        logger=True,
    )

    log(
        f"{mode}_precision",
        precision,
        on_step=False,
        on_epoch=True,
        logger=True,
    )


class Averager:
    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v


def save_results(results_dir, model_name, results):
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    filename = os.path.join(results_dir, f"{model_name}_results.json")

    with open(filename, "w") as f:
        json.dump(results, f)
