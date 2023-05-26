from models.shared.layers import (
    MLP,
    MaskAttention,
    SelfAttentionFeatureExtract,
    cnn_extractor,
)
import torch
import torch.nn as nn
from sklearn.metrics import *


class MultiDomainFENDModel(torch.nn.Module):
    def __init__(self, emb_dim, mlp_dims, domain_num, dropout, bert):
        super(MultiDomainFENDModel, self).__init__()
        self.domain_num = domain_num
        self.gamma = 10
        self.num_expert = 5
        self.fea_size = 256
        self.bert = bert

        feature_kernel = {1: 64, 2: 64, 3: 64, 5: 64, 10: 64}
        expert = []
        for i in range(self.num_expert):
            expert.append(cnn_extractor(feature_kernel, emb_dim))
        self.expert = nn.ModuleList(expert)

        self.gate = nn.Sequential(
            nn.Linear(emb_dim, mlp_dims[-1]),
            nn.ReLU(),
            nn.Linear(mlp_dims[-1], self.num_expert),
            nn.Softmax(dim=1),
        )

        self.attention = MaskAttention(emb_dim)

        self.domain_embedder = nn.Embedding(
            num_embeddings=self.domain_num, embedding_dim=emb_dim
        )
        self.specific_extractor = SelfAttentionFeatureExtract(
            multi_head_num=1, input_size=emb_dim, output_size=self.fea_size
        )
        self.classifier = MLP(320, mlp_dims, dropout)

    def forward(self, **kwargs):
        inputs = kwargs["content"]
        masks = kwargs["content_masks"]
        category = kwargs["category"]
        init_feature = self.bert(inputs, attention_mask=masks).last_hidden_state

        feature, _ = self.attention(init_feature, masks)
        idxs = torch.tensor([index for index in category]).view(-1, 1).cuda()
        domain_embedding = self.domain_embedder(idxs).squeeze(1)

        gate_value = self.gate(domain_embedding)

        shared_feature = 0
        for i in range(self.num_expert):
            tmp_feature = self.expert[i](init_feature)
            shared_feature += tmp_feature * gate_value[:, i].unsqueeze(1)

        label_pred = self.classifier(shared_feature)

        return torch.sigmoid(label_pred.squeeze(1))
