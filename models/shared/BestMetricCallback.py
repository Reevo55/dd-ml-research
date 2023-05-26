import os
from pytorch_lightning.callbacks import Callback
import torch


class BestMetricCallback(Callback):
    def __init__(self, save_dir, model_name):
        super().__init__()
        self.save_dir = save_dir
        self.model_name = model_name
        self.best_metric = None
        self.best_mem = None

    def on_validation_epoch_end(self, trainer, pl_module):
        current_metric = trainer.callback_metrics["val_loss"]
        if self.best_metric is None or self.best_metric > current_metric:
            self.best_metric = current_metric
            torch.save(
                pl_module.model.state_dict(),
                os.path.join(self.save_dir, f"parameter_{self.model_name}.pkl"),
            )

    def on_train_end(self, trainer, pl_module):
        pl_module.model.load_state_dict(
            torch.load(os.path.join(self.save_dir, f"parameter_{self.model_name}.pkl"))
        )
