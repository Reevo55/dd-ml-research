import torch
import torch.nn as nn
import numpy as np
from ..shared.layers import *
from sklearn.metrics import *
import math
from sklearn.cluster import KMeans
import numpy as np


def cal_length(x):
    return torch.sqrt(torch.sum(torch.pow(x, 2), dim=1))


def norm(x):
    length = cal_length(x).view(-1, 1)
    x = x / length
    return x


def convert_to_onehot(label, batch_size, num):
    return torch.zeros(batch_size, num).cuda().scatter_(1, label, 1)


class MemoryNetwork(torch.nn.Module):
    def __init__(self, input_dim, emb_dim, domain_num, memory_num=10):
        super(MemoryNetwork, self).__init__()
        self.domain_num = domain_num
        self.emb_dim = emb_dim
        self.memory_num = memory_num
        self.tau = 32
        self.topic_fc = torch.nn.Linear(input_dim, emb_dim, bias=False)
        self.domain_fc = torch.nn.Linear(input_dim, emb_dim, bias=False)

        self.domain_memory = dict()

    def forward(self, feature, category):
        feature = norm(feature)
        domain_label = torch.tensor([index for index in category]).view(-1, 1).cuda()
        domain_memory = []
        for i in range(self.domain_num):
            domain_memory.append(self.domain_memory[i])

        sep_domain_embedding = []
        for i in range(self.domain_num):
            topic_att = torch.nn.functional.softmax(
                torch.mm(self.topic_fc(feature), domain_memory[i].T) * self.tau, dim=1
            )
            tmp_domain_embedding = torch.mm(topic_att, domain_memory[i])
            sep_domain_embedding.append(tmp_domain_embedding.unsqueeze(1))
        sep_domain_embedding = torch.cat(sep_domain_embedding, 1)

        domain_att = torch.bmm(
            sep_domain_embedding, self.domain_fc(feature).unsqueeze(2)
        ).squeeze()

        domain_att = torch.nn.functional.softmax(
            domain_att * self.tau, dim=1
        ).unsqueeze(1)

        return domain_att

    def write(self, all_feature, category):
        domain_fea_dict = {}
        domain_set = set(category.cpu().detach().numpy().tolist())
        for i in domain_set:
            domain_fea_dict[i] = []
        for i in range(all_feature.size(0)):
            domain_fea_dict[category[i].item()].append(all_feature[i].view(1, -1))

        for i in domain_set:
            domain_fea_dict[i] = torch.cat(domain_fea_dict[i], 0)
            topic_att = torch.nn.functional.softmax(
                torch.mm(self.topic_fc(domain_fea_dict[i]), self.domain_memory[i].T)
                * self.tau,
                dim=1,
            ).unsqueeze(2)
            tmp_fea = domain_fea_dict[i].unsqueeze(1).repeat(1, self.memory_num, 1)
            new_mem = tmp_fea * topic_att
            new_mem = new_mem.mean(dim=0)
            topic_att = torch.mean(topic_att, 0).view(-1, 1)
            self.domain_memory[i] = (
                self.domain_memory[i]
                - 0.05 * topic_att * self.domain_memory[i]
                + 0.05 * new_mem
            )


class M3FENDv2Normal(torch.nn.Module):
    def __init__(
        self,
        emb_dim,
        mlp_dims,
        dropout,
        semantic_num,
        emotion_num,
        style_num,
        LNN_dim,
        domain_num,
        bert,
    ):
        super(M3FENDv2Normal, self).__init__()
        self.domain_num = domain_num
        self.gamma = 10
        self.memory_num = 10
        self.semantic_num_expert = semantic_num
        self.emotion_num_expert = emotion_num
        self.style_num_expert = style_num
        self.LNN_dim = LNN_dim
        print(
            "semantic_num_expert:",
            self.semantic_num_expert,
            "emotion_num_expert:",
            self.emotion_num_expert,
            "style_num_expert:",
            self.style_num_expert,
            "lnn_dim:",
            self.LNN_dim,
        )
        self.fea_size = 256
        self.emb_dim = emb_dim
        self.bert = bert

        feature_kernel = {1: 64, 2: 64, 3: 64, 5: 64, 10: 64}

        content_expert = []
        for i in range(self.semantic_num_expert):
            content_expert.append(cnn_extractor(feature_kernel, emb_dim))
        self.content_expert = nn.ModuleList(content_expert)

        emotion_expert = []
        for i in range(self.emotion_num_expert):
            emotion_expert.append(
                MLP(
                    38 * 5,
                    [
                        256,
                        320,
                    ],
                    dropout,
                    output_layer=False,
                )
            )
        self.emotion_expert = nn.ModuleList(emotion_expert)

        style_expert = []
        for i in range(self.style_num_expert):
            style_expert.append(
                MLP(
                    32,
                    [
                        256,
                        320,
                    ],
                    dropout,
                    output_layer=False,
                )
            )
        self.style_expert = nn.ModuleList(style_expert)

        self.gate = nn.Sequential(
            nn.Linear(self.emb_dim * 2, mlp_dims[-1]),
            nn.GELU(),
            nn.Linear(mlp_dims[-1], self.LNN_dim),
            nn.Softmax(dim=1),
        )

        self.attention = MaskAttention(emb_dim)

        self.weight = (
            torch.nn.Parameter(
                torch.Tensor(
                    self.LNN_dim,
                    self.semantic_num_expert
                    + self.emotion_num_expert
                    + self.style_num_expert,
                )
            )
            .unsqueeze(0)
            .cuda()
        )
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

        self.domain_memory = MemoryNetwork(
            input_dim=self.emb_dim + 38 * 5 + 32,
            emb_dim=self.emb_dim + 38 * 5 + 32,
            domain_num=self.domain_num,
            memory_num=self.memory_num,
        )

        self.domain_embedder = nn.Embedding(
            num_embeddings=self.domain_num, embedding_dim=emb_dim
        )
        self.all_feature = {}

        self.classifier = MLP(320, mlp_dims, dropout)

    def forward(self, **kwargs):
        content = kwargs["content"]
        content_masks = kwargs["content_masks"]

        content_emotion = kwargs["content_emotion"]
        comments_emotion = kwargs["comments_emotion"]
        emotion_gap = kwargs["emotion_gap"]
        style_feature = kwargs["style_feature"]
        emotion_feature = torch.cat(
            [content_emotion, comments_emotion, emotion_gap], dim=1
        )
        category = kwargs["category"]

        content_feature = self.bert(content, attention_mask=content_masks)[0]

        gate_input_feature, _ = self.attention(content_feature, content_masks)
        memory_att = self.domain_memory(
            torch.cat([gate_input_feature, emotion_feature, style_feature], dim=-1),
            category,
        )
        domain_emb_all = self.domain_embedder(
            torch.LongTensor(range(self.domain_num)).cuda()
        )
        general_domain_embedding = torch.mm(memory_att.squeeze(1), domain_emb_all)

        idxs = torch.tensor([index for index in category]).view(-1, 1).cuda()
        domain_embedding = self.domain_embedder(idxs).squeeze(1)
        gate_input = torch.cat([domain_embedding, general_domain_embedding], dim=-1)

        gate_value = self.gate(gate_input).view(
            content_feature.size(0), 1, self.LNN_dim
        )

        shared_feature = []
        for i in range(self.semantic_num_expert):
            shared_feature.append(self.content_expert[i](content_feature).unsqueeze(1))

        for i in range(self.emotion_num_expert):
            shared_feature.append(self.emotion_expert[i](emotion_feature).unsqueeze(1))

        for i in range(self.style_num_expert):
            shared_feature.append(self.style_expert[i](style_feature).unsqueeze(1))

        shared_feature = torch.cat(shared_feature, dim=1)

        embed_x_abs = torch.abs(shared_feature)
        embed_x_afn = torch.add(embed_x_abs, 1e-7)
        embed_x_log = torch.log1p(embed_x_afn)

        lnn_out = torch.matmul(self.weight, embed_x_log)
        lnn_exp = torch.expm1(lnn_out)
        shared_feature = lnn_exp.contiguous().view(-1, self.LNN_dim, 320)

        shared_feature = torch.bmm(gate_value, shared_feature).squeeze()

        deep_logits = self.classifier(shared_feature)

        return torch.sigmoid(deep_logits.squeeze(1))

    def save_feature(self, **kwargs):
        content = kwargs["content"]
        content_masks = kwargs["content_masks"]

        content_emotion = kwargs["content_emotion"]
        comments_emotion = kwargs["comments_emotion"]
        emotion_gap = kwargs["emotion_gap"]
        emotion_feature = torch.cat(
            [content_emotion, comments_emotion, emotion_gap], dim=1
        )

        style_feature = kwargs["style_feature"]

        category = kwargs["category"]

        content_feature = self.bert(content, attention_mask=content_masks)[0]
        content_feature, _ = self.attention(content_feature, content_masks)

        all_feature = torch.cat(
            [content_feature, emotion_feature, style_feature], dim=1
        )
        all_feature = norm(all_feature)

        for index in range(all_feature.size(0)):
            domain = int(category[index].cpu().numpy())
            if not (domain in self.all_feature):
                self.all_feature[domain] = []
            self.all_feature[domain].append(
                all_feature[index].view(1, -1).cpu().detach().numpy()
            )

    def init_memory(self):
        for domain in self.all_feature:
            all_feature = np.concatenate(self.all_feature[domain])
            kmeans = KMeans(n_clusters=self.memory_num, init="k-means++").fit(
                all_feature
            )
            centers = kmeans.cluster_centers_
            centers = torch.from_numpy(centers).cuda()
            self.domain_memory.domain_memory[domain] = centers

    def write(self, **kwargs):
        content = kwargs["content"]
        content_masks = kwargs["content_masks"]

        content_emotion = kwargs["content_emotion"]
        comments_emotion = kwargs["comments_emotion"]
        emotion_gap = kwargs["emotion_gap"]
        emotion_feature = torch.cat(
            [content_emotion, comments_emotion, emotion_gap], dim=1
        )

        style_feature = kwargs["style_feature"]

        category = kwargs["category"]

        content_feature = self.bert(content, attention_mask=content_masks)[0]
        content_feature, _ = self.attention(content_feature, content_masks)

        all_feature = torch.cat(
            [content_feature, emotion_feature, style_feature], dim=1
        )
        all_feature = norm(all_feature)
        self.domain_memory.write(all_feature, category)
