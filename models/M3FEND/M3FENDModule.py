import os
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
from torch.nn import BCELoss
import tqdm

from .M3FEND import M3FENDNormal
from utils.utils import category_logs, data2gpu, default_log
from torch.nn import BCELoss
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR


class M3FENDModule(LightningModule):
    def __init__(
        self,
        emb_dim,
        mlp_dims,
        lr,
        dropout,
        category_dict,
        weight_decay,
        save_param_dir,
        bert,
        train_loader,
        use_cuda=torch.cuda.is_available(),
        semantic_num=7,
        emotion_num=7,
        style_num=2,
        lnn_dim=50,
    ):
        super().__init__()

        self.lr = lr
        self.weight_decay = weight_decay
        self.use_cuda = use_cuda
        self.category_dict = category_dict
        self.emb_dim = emb_dim
        self.mlp_dims = mlp_dims
        self.dropout = dropout
        self.semantic_num = semantic_num
        self.emotion_num = emotion_num
        self.style_num = style_num
        self.lnn_dim = lnn_dim
        self.bert = bert
        self.train_loader = train_loader
        self.save_param_dir = save_param_dir

        self.model = M3FENDNormal(
            self.emb_dim,
            self.mlp_dims,
            self.dropout,
            self.semantic_num,
            self.emotion_num,
            self.style_num,
            self.lnn_dim,
            len(self.category_dict),
            self.bert,
        )
        self.model.cuda()
        self.loss_fn = BCELoss()

        self.model.train()
        train_data_iter = tqdm.tqdm(self.train_loader)

        for step_n, batch in enumerate(train_data_iter):
            batch_data = data2gpu(batch, self.use_cuda)
            label_pred = self.model.save_feature(**batch_data)
        self.model.init_memory()

        if os.path.exists(save_param_dir):
            self.save_param_dir = save_param_dir
        else:
            self.save_param_dir = save_param_dir
            os.makedirs(save_param_dir)

    def forward(self, x):
        return self.model(x)

    def _log(self, mode, label, label_pred, loss):
        category_logs(
            self.log, self.category_dict, mode, label, label_pred, loss, self.loss_fn
        )

    def _step(self, batch, batch_idx, mode):
        batch, label, label_pred, loss = self._intro_to_training_step(batch, batch_idx)
        self._log(mode, label, label_pred, loss)
        return loss

    def _intro_to_training_step(self, batch, batch_idx):
        batch_data = data2gpu(batch, self.use_cuda)
        label = batch_data["label"]
        category = batch_data["category"]
        label_pred = self.model(**batch_data)
        loss = self.loss_fn(label_pred, label.float())
        return batch_data, label, label_pred, loss

    def training_step(self, batch, batch_idx):
        batch_data, label, label_pred, loss = self._intro_to_training_step(
            batch, batch_idx
        )
        with torch.no_grad():
            self.model.write(**batch_data)
        self._log("training", label, label_pred, loss)
        return loss

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "test")

    def configure_optimizers(self):
        optimizer = Adam(
            params=self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = StepLR(optimizer, step_size=100, gamma=0.98)
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "train_loss",
        }
