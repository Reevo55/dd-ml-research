import torch
import random
import pandas as pd
import tqdm
import numpy as np
import pickle
import re
from transformers import BertTokenizer
from transformers import RobertaTokenizer
from torch.utils.data import TensorDataset, DataLoader


def _init_fn(worker_id):
    np.random.seed(2021)


def read_pkl(path):
    with open(path, "rb") as f:
        t = pickle.load(f)
    return t


def df_filter(df_data, category_dict):
    df_data = df_data[df_data["category"].isin(set(category_dict.keys()))]
    return df_data


def word2input(texts, max_len, tokenizer):
    encoding = tokenizer(
        texts.tolist(),
        pad_token="[PAD]",
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_len,
    )
    token_ids = encoding["input_ids"]
    masks = encoding["attention_mask"]
    return token_ids, masks


def process(x):
    x["content_emotion"] = x["content_emotion"].astype(float)
    return x


class MyDataloader:
    def __init__(
        self,
        max_len,
        batch_size,
        category_dict,
        num_workers=2,
        subset_size=100_000,
        tokenizer=RobertaTokenizer.from_pretrained("roberta-base"),
    ):
        self.max_len = max_len
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.category_dict = category_dict
        self.tokenizer = tokenizer
        self.subset_size = subset_size

    def load_data(self, path, shuffle):
        print("Loading data from {}".format(path))
        self.data = df_filter(read_pkl(path), self.category_dict)

        if self.subset_size is not None and self.subset_size < len(self.data):
            self.data = self.data.sample(n=self.subset_size, random_state=42)

        content = self.data["content"].to_numpy()
        comments = self.data["comments"].to_numpy()
        content_emotion = torch.tensor(
            np.vstack(self.data["content_emotion"]).astype("float32")
        )
        comments_emotion = torch.tensor(
            np.vstack(self.data["comments_emotion"]).astype("float32")
        )
        emotion_gap = torch.tensor(
            np.vstack(self.data["emotion_gap"]).astype("float32")
        )
        style_feature = torch.tensor(
            np.vstack(self.data["style_feature"]).astype("float32")
        )
        label = torch.tensor(self.data["label"].astype(int).to_numpy())
        category = torch.tensor(
            self.data["category"].apply(lambda c: self.category_dict[c]).to_numpy()
        )
        content_token_ids, content_masks = word2input(
            content, self.max_len, self.tokenizer
        )
        comments_token_ids, comments_masks = word2input(
            comments, self.max_len, self.tokenizer
        )
        dataset = TensorDataset(
            content_token_ids,
            content_masks,
            comments_token_ids,
            comments_masks,
            content_emotion,
            comments_emotion,
            emotion_gap,
            style_feature,
            label,
            category,
        )
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=shuffle,
            worker_init_fn=_init_fn,
        )
        return dataloader
