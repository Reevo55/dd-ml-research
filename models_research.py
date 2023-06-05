import pytorch_lightning as pl
from transformers import RobertaTokenizer
from factories.ModelFactory import ModelFactory

import torch
from pytorch_lightning.loggers import TensorBoardLogger
from transformers import RobertaModel
from dataloader.MyDataloader import MyDataloader
from utils.utils import save_results

torch.set_float32_matmul_precision("medium")

input_dim = 100
emb_dim = 768
mlp_dims = [512, 256]
lr = 0.0001
dropout = 0.2
weight_decay = 0.0001
save_param_dir = "./params"
max_len = 170
epochs = 50

batch_size = 64
subset_size = 128
subset_size = None

category_dict = {
    "gossipcop": 0,
    "politifact": 1,
    "COVID": 2,
}
num_workers = 3

train_path = "./data/en/train.pkl"
val_path = "./data/en/val.pkl"
test_path = "./data/en/test.pkl"

results_dir = "./results"


if __name__ == "__main__":
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    bert = RobertaModel.from_pretrained("roberta-base").requires_grad_(False).to("cuda")

    loader = MyDataloader(
        max_len=max_len,
        batch_size=batch_size,
        subset_size=subset_size,
        category_dict=category_dict,
        num_workers=num_workers,
        tokenizer=tokenizer,
    )

    train_loader = loader.load_data(train_path, True)
    val_loader = loader.load_data(val_path, False)
    test_loader = loader.load_data(test_path, False)

    models = ["M3FEND", "MDFEND", "DualEmotion", "moSEM"]
    all_results = {}

    for model_name in models:
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

        callbacks = []
        if callback is not None:
            callbacks.append(callback)

        logger = TensorBoardLogger(
            save_dir="logs", name="model_research", version=model_name
        )
        trainer = pl.Trainer(
            max_epochs=epochs, accelerator="gpu", logger=logger, callbacks=callbacks
        )
        trainer.fit(model, train_loader, val_loader)

        result = trainer.test(model, dataloaders=test_loader)

        print(f"Results for {model_name}:", result[0])

        all_results[model_name] = result[0]

    save_results("results", "research_results", all_results)
