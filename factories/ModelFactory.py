from models.DualEmotion.DualEmotionModule import DualEmotionModel
from models.M3FEND.BestModelMemoryCallback import BestModelMemoryCallback
from models.M3FEND.M3FENDModule import M3FENDModule
from models.MDFEND.MDFENDModule import MDFENDModule
from models.shared.BestMetricCallback import BestMetricCallback
from models.mose.MoseModule import MoSEModule


class ModelFactory:
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
    ):
        self.emb_dim = emb_dim
        self.mlp_dims = mlp_dims
        self.lr = lr
        self.dropout = dropout
        self.category_dict = category_dict
        self.weight_decay = weight_decay
        self.save_param_dir = save_param_dir
        self.bert = bert
        self.train_loader = train_loader

    def create_model(self, model):
        if model == "M3FEND":
            return M3FENDModule(
                emb_dim=self.emb_dim,
                mlp_dims=self.mlp_dims,
                lr=self.lr,
                dropout=self.dropout,
                category_dict=self.category_dict,
                weight_decay=self.weight_decay,
                save_param_dir=self.save_param_dir,
                bert=self.bert,
                train_loader=self.train_loader,
            ), BestModelMemoryCallback(
                save_dir=self.save_param_dir, model_name="M3FEND"
            )

        elif model == "MDFEND":
            return (
                MDFENDModule(
                    emb_dim=self.emb_dim,
                    mlp_dims=self.mlp_dims,
                    lr=self.lr,
                    dropout=self.dropout,
                    category_dict=self.category_dict,
                    weight_decay=self.weight_decay,
                    save_param_dir=self.save_param_dir,
                    bert=self.bert,
                ),
                BestMetricCallback(save_dir=self.save_param_dir, model_name="MDFEND"),
            )

        elif model == "DualEmotion":
            return (
                DualEmotionModel(
                    emb_dim=self.emb_dim,
                    mlp_dims=self.mlp_dims,
                    lr=self.lr,
                    dropout=self.dropout,
                    category_dict=self.category_dict,
                    weight_decay=self.weight_decay,
                    save_param_dir=self.save_param_dir,
                    bert=self.bert,
                ),
                BestMetricCallback(
                    save_dir=self.save_param_dir, model_name="DualEmotion"
                ),
            )

        elif model == "moSEM":
            return (
                MoSEModule(
                    emb_dim=self.emb_dim,
                    mlp_dims=self.mlp_dims,
                    lr=self.lr,
                    dropout=self.dropout,
                    category_dict=self.category_dict,
                    weight_decay=self.weight_decay,
                    save_param_dir=self.save_param_dir,
                    bert=self.bert,
                ),
                BestMetricCallback(
                    save_dir=self.save_param_dir, model_name="DualEmotion"
                ),
            )
