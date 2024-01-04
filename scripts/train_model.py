import os, sys, torch
sys.path.append(os.getcwd())
from lightning.pytorch.loggers import WandbLogger
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint

# choose model and data_module --------------------------------
from src.data_modules import MovingImpulseResponseDataModule
from src.cnn_model import cnn_model

# define model, (from scratch or checkpoint)
#model = cnn_model(scale_model_width=10) # from scratch
model = cnn_model.load_from_checkpoint("checkpoints/epoch=86-step=13224.ckpt", scale_model_width=5, map_location=torch.device("cpu")) # from check point

data_path = "./data/datasets/moving_dataset_medium.hdf5"
sound_dir = "./data/reference_data/reference_sounds/"
data_module = MovingImpulseResponseDataModule(data_path,sound_dir,batch_size=20)
#-----------------------------------------------
wandb_logger = WandbLogger(log_model="all")
checkpoint_callback = ModelCheckpoint(every_n_epochs=50)
trainer = L.Trainer(
    max_epochs=1000,
    accelerator="cuda",
    devices=[1],
    logger=wandb_logger,
    log_every_n_steps=50,
    callbacks=[checkpoint_callback],
)
trainer.fit(model, data_module)