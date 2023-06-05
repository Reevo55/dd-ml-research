from pytorch_lightning.callbacks import EarlyStopping

early_stop_callback = EarlyStopping(
    monitor="val_loss", min_delta=0.00, patience=5, verbose=False, mode="min"
)
