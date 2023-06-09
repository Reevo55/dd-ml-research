import optuna
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from optuna.integration import PyTorchLightningPruningCallback

import pytorch_lightning as pl
from factories.ModelFactory import ModelFactory
from transformers import (
    DebertaTokenizer,
    DebertaModel,
)
import torch
import torch.multiprocessing as mp
from pytorch_lightning.loggers import TensorBoardLogger
from transformers import RobertaConfig, RobertaModel

from dataloader.MyDataloader import MyDataloader
from utils.utils import save_results
from pytorch_lightning.callbacks import EarlyStopping

torch.set_float32_matmul_precision("medium")

input_dim = 100
emb_dim = 768
mlp_dims = [512, 256]
lr = 0.0001
dropout = 0.2
weight_decay = 0.001
save_param_dir = "./params"
max_len = 170
# epochs = 50
epochs = 15

batch_size = 64
subset_size = 128
subset_size = 128 * 32
category_dict = {
    "gossipcop": 0,
    "politifact": 1,
    "COVID": 2,
}
num_workers = 3

train_path = "./data/en/train.pkl"
val_path = "./data/en/val.pkl"
test_path = "./data/en/test.pkl"


def objective(trial):
    # Defining the hyperparameters to tune
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-1)
    dropout = trial.suggest_uniform("dropout", 0.1, 0.5)

    model_name = "M3FENDv2"
    model, callback = ModelFactory(
        emb_dim=emb_dim,
        mlp_dims=mlp_dims,
        lr=lr,
        dropout=dropout,
        category_dict=category_dict,
        weight_decay=weight_decay,
        save_param_dir=save_param_dir,
        bert=bert,
        train_loader=train_loader,
    ).create_model(model_name)

    # Set PyTorch Lightning Callbacks

    callbacks = []

    if callback is not None:
        callbacks.append(callback)

    logger = TensorBoardLogger(save_dir="logs", name="hyperparams", version=model_name)

    early_stop_callback = EarlyStopping(
        monitor="val_loss", min_delta=0.00, patience=2, verbose=False, mode="min"
    )
    callbacks.append(early_stop_callback)

    # Include Optuna pruning callback
    callbacks.append(PyTorchLightningPruningCallback(trial, monitor="val_loss"))

    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="gpu",
        logger=logger,
        callbacks=callbacks,
    )

    trainer.fit(model, train_loader, val_loader)

    result = trainer.test(model, dataloaders=test_loader)

    return result[0]["test_loss"]


if __name__ == "__main__":
    tokenizer = DebertaTokenizer.from_pretrained("microsoft/deberta-base")

    bert = (
        DebertaModel.from_pretrained("microsoft/deberta-base")
        .requires_grad_(False)
        .to("cuda")
    )

    loader = MyDataloader(
        max_len=max_len,
        batch_size=batch_size,
        subset_size=subset_size,
        category_dict=category_dict,
        num_workers=num_workers,
        tokenizer=tokenizer,
    )

    train_loader = loader.load_data(train_path, True)
    val_loader = loader.load_data(val_path, True)
    test_loader = loader.load_data(test_path, True)

    study = optuna.create_study(
        direction="minimize", pruner=optuna.pruners.MedianPruner()
    )
    study.optimize(objective, n_trials=50, timeout=3600 * 8)

    print("Best trial:", study.best_trial.params)
