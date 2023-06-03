import pytorch_lightning as pl
import torch
from transformers import (
    XLMRobertaTokenizer,
    XLMRobertaModel,
    BigBirdTokenizer,
    BigBirdModel,
    ElectraTokenizer,
    ElectraModel,
    DebertaTokenizer,
    DebertaModel,
    AlbertTokenizer,
    AlbertModel,
)
from factories.ModelFactory import ModelFactory
from dataloader.MyDataloader import MyDataloader
from pytorch_lightning.loggers import TensorBoardLogger

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
epochs = 2

batch_size = 128
subset_size = 128
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


def perform_research(model_name, tokenizer, bert):
    tokenizer_name = tokenizer.__class__.__name__
    bert_name = bert.__class__.__name__

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
        save_dir="logs",
        name="embeddings",
        version=f"{model_name}_{tokenizer_name}_{bert_name}",
    )
    trainer = pl.Trainer(
        max_epochs=epochs, accelerator="gpu", logger=logger, callbacks=callbacks
    )
    trainer.fit(model, train_loader, val_loader)

    result = trainer.test(model, dataloaders=test_loader)

    print(f"Results for {model_name} with {tokenizer_name} and {bert_name}:", result[0])

    # Save results to dictionary
    return (model_name, tokenizer_name, bert_name), result[0]


if __name__ == "__main__":
    # Define the models, tokenizers, and BERT models to research
    models = ["M3FEND", "MDFEND", "DualEmotion", "moSEM"]
    tokenizers = [
        BigBirdTokenizer.from_pretrained("google/bigbird-roberta-base"),
        ElectraTokenizer.from_pretrained("google/electra-base-discriminator"),
        XLMRobertaTokenizer.from_pretrained("xlm-roberta-base"),
        DebertaTokenizer.from_pretrained("microsoft/deberta-base"),
        AlbertTokenizer.from_pretrained("albert-base-v2"),
    ]
    bert_models = [
        BigBirdModel.from_pretrained("google/bigbird-roberta-base")
        .requires_grad_(False)
        .to("cuda"),
        ElectraModel.from_pretrained("google/electra-base-discriminator")
        .requires_grad_(False)
        .to("cuda"),
        XLMRobertaModel.from_pretrained("xlm-roberta-base")
        .requires_grad_(False)
        .to("cuda"),
        DebertaModel.from_pretrained("microsoft/deberta-base")
        .requires_grad_(False)
        .to("cuda"),
        AlbertModel.from_pretrained("albert-base-v2").requires_grad_(False).to("cuda"),
    ]

    all_results = {}

    for model_name in models:
        for i in range(len(tokenizers)):
            key, result = perform_research(model_name, tokenizers[i], bert_models[i])
            all_results[key] = result

    # Save results as JSON
    save_results("research_results", all_results)
