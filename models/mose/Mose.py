import torch
import torch.nn as nn
from sklearn.metrics import *
from ..shared.layers import *


class MoSEM(torch.nn.Module):
    def __init__(self, emb_dim, mlp_dims, num_head, bert, num_layers=1):
        super(MoSEM, self).__init__()
        self.num_expert = 5
        self.num_head = num_head
        self.fea_size = 320
        self.bert = bert

        expert = []
        for i in range(self.num_expert):
            expert.append(
                torch.nn.Sequential(
                    nn.LSTM(
                        input_size=self.fea_size,
                        hidden_size=self.fea_size,
                        num_layers=num_layers,
                        batch_first=True,
                        bidirectional=True,
                    )
                )
            )
        self.expert = nn.ModuleList(expert)

        mask = []
        for i in range(self.num_expert):
            mask.append(MaskAttention(self.fea_size * 2))
        self.mask = nn.ModuleList(mask)

        head = []
        for i in range(self.num_head):
            head.append(torch.nn.Linear(self.fea_size * 2, 1))
        self.head = nn.ModuleList(head)

        gate = []
        for i in range(self.num_head):
            gate.append(
                torch.nn.Sequential(
                    torch.nn.Linear(emb_dim, mlp_dims[-1]),
                    torch.nn.ReLU(),
                    torch.nn.Linear(mlp_dims[-1], self.num_expert),
                    torch.nn.Softmax(dim=1),
                )
            )
        self.gate = nn.ModuleList(gate)

        self.rnn = nn.LSTM(
            input_size=emb_dim,
            hidden_size=self.fea_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
        )

        self.attention = MaskAttention(emb_dim)

    def forward(self, **kwargs):
        inputs = kwargs["content"]
        masks = kwargs["content_masks"]
        category = kwargs["category"]
        feature = self.bert(inputs, attention_mask=masks)[0]

        gate_feature, _ = self.attention(feature)
        gate_value = []
        for i in range(feature.size(0)):
            gate_value.append(self.gate[category[i]](gate_feature[i].view(1, -1)))
        gate_value = torch.cat(gate_value)

        feature, _ = self.rnn(feature)

        rep = 0
        for i in range(self.num_expert):
            tmp_fea, _ = self.expert[i](feature)
            tmp_fea, _ = self.mask[i](tmp_fea, masks)
            rep += gate_value[:, i].unsqueeze(1) * tmp_fea

        output = []
        for i in range(feature.size(0)):
            output.append(self.head[category[i]](rep[i].view(1, -1)))
        output = torch.cat(output)

        return torch.sigmoid(output.squeeze(1))
