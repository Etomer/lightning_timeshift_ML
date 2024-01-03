import os, sys
sys.path.append(os.getcwd())
from pytorch_lightning.loggers import WandbLogger
import lightning as L

# choose model and data_module --------------------------------
from src.data_modules import MovingImpulseResponseDataModule
from src.cnn_model import cnn_model
model = cnn_model()
data_path = "./data/datasets/moving_dataset_small.hdf5"
sound_dir = "./data/reference_data/reference_sounds/"
data_module = MovingImpulseResponseDataModule(data_path,sound_dir,batch_size=20)
#-----------------------------------------------

wandb_logger = WandbLogger(log_model="all")
trainer = L.Trainer(
    max_epochs=1000,
    accelerator="cuda",
    devices=1,
    logger=wandb_logger,
    log_every_n_steps=1,
)
trainer.fit(model, data_module)