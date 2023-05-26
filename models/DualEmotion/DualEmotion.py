from ..shared.layers import *
import torch
import torch.nn as nn
from sklearn.metrics import *


class DualEmotion(torch.nn.Module):
    def __init__(self, emb_dim, mlp_dims, dropout, bert):
        super(DualEmotion, self).__init__()
        self.fea_size = emb_dim
        self.bert = bert

        self.rnn = nn.GRU(
            input_size=emb_dim,
            hidden_size=self.fea_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.attention = MaskAttention(self.fea_size * 2)
        self.classifier = MLP(self.fea_size * 2 + 38 * 5, mlp_dims, dropout)

    def forward(self, **kwargs):
        content = kwargs["content"]
        content_masks = kwargs["content_masks"]
        content_emotion = kwargs["content_emotion"]
        comments_emotion = kwargs["comments_emotion"]
        emotion_gap = kwargs["emotion_gap"]
        emotion_feature = torch.cat(
            [content_emotion, comments_emotion, emotion_gap], dim=1
        )

        content_feature = self.bert(content, attention_mask=content_masks)[0]
        content_feature, _ = self.rnn(content_feature)
        content_feature, _ = self.attention(content_feature, content_masks)

        shared_feature = torch.cat([content_feature, emotion_feature], dim=1)

        label_pred = self.classifier(shared_feature)

        return torch.sigmoid(label_pred.squeeze(1))
