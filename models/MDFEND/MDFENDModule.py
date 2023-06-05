import os
import torch
from pytorch_lightning import LightningModule
from torch.optim.lr_scheduler import StepLR
from torch.nn import BCELoss
from torch.optim import Adam

from models.MDFEND.MDFEND import MultiDomainFENDModel
from utils.utils import category_logs, data2gpu, default_log


class MDFENDModule(LightningModule):
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
        use_cuda=torch.cuda.is_available(),
    ):
        super().__init__()

        self.lr = lr
        self.weight_decay = weight_decay
        self.category_dict = category_dict
        self.emb_dim = emb_dim
        self.mlp_dims = mlp_dims
        self.dropout = dropout
        self.use_cuda = use_cuda
        self.bert = bert
        self.validation_step_output = []
        self.train_step_output = []
        self.test_step_output = []
        self.model = MultiDomainFENDModel(
            self.emb_dim,
            self.mlp_dims,
            len(self.category_dict),
            self.dropout,
            self.bert,
        )

        self.loss_fn = BCELoss()

        if os.path.exists(save_param_dir):
            self.save_param_dir = save_param_dir
        else:
            self.save_param_dir = save_param_dir
            os.makedirs(save_param_dir)

    def forward(self, x):
        return self.model(x)

    def _log(self, mode, label, label_pred, loss, category):
        category_logs(
            self.log,
            self.category_dict,
            mode,
            label,
            label_pred,
            loss,
            self.loss_fn,
            category,
        )

    def _step(self, batch, batch_idx, mode):
        batch_data, label, label_pred, loss, category = self._intro_to_training_step(
            batch, batch_idx
        )
        return {
            "loss": loss,
            "label": label,
            "label_pred": label_pred,
            "category": category,
        }

    def _intro_to_training_step(self, batch, batch_idx):
        batch_data = data2gpu(batch, self.use_cuda)
        label = batch_data["label"]
        category = batch_data["category"]
        label_pred = self.model(**batch_data)
        loss = self.loss_fn(label_pred, label.float())
        return batch_data, label, label_pred, loss, category

    def training_step(self, batch, batch_idx):
        output = self._step(batch, batch_idx, "training")

        self.train_step_output.append(output)
        return output

    def validation_step(self, batch, batch_idx):
        output = self._step(batch, batch_idx, "val")
        self.log("val_loss", output["loss"])
        self.validation_step_output.append(output)
        return output

    def test_step(self, batch, batch_idx):
        output = self._step(batch, batch_idx, "test")

        self.test_step_output.append(output)
        return output

    def on_train_epoch_end(self):
        label = torch.cat([x["label"] for x in self.train_step_output])
        label_pred = torch.cat([x["label_pred"] for x in self.train_step_output])
        avg_loss = torch.stack([x["loss"] for x in self.train_step_output]).mean()
        category = torch.cat(
            [x["category"] for x in self.train_step_output]
        )  # Get the category from the outputs

        self._log("training", label, label_pred, avg_loss, category)

    def on_validation_epoch_end(self):
        label = torch.cat([x["label"] for x in self.validation_step_output])
        label_pred = torch.cat([x["label_pred"] for x in self.validation_step_output])
        avg_loss = torch.stack([x["loss"] for x in self.validation_step_output]).mean()
        category = torch.cat(
            [x["category"] for x in self.validation_step_output]
        )  # Get the category from the outputs

        self._log("val", label, label_pred, avg_loss, category)

    def on_test_epoch_end(self):
        label = torch.cat([x["label"] for x in self.test_step_output])
        label_pred = torch.cat([x["label_pred"] for x in self.test_step_output])
        avg_loss = torch.stack([x["loss"] for x in self.test_step_output]).mean()
        category = torch.cat(
            [x["category"] for x in self.test_step_output]
        )  # Get the category from the outputs

        self._log("test", label, label_pred, avg_loss, category)

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
