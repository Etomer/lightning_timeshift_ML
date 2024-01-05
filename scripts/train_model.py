import os, sys, torch
sys.path.append(os.getcwd())
from lightning.pytorch.loggers import WandbLogger
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from torchvision import transforms

# choose model and data_module --------------------------------
from src.data_modules import MovingImpulseResponseDataModule
from src.cnn_model import cnn_model
import src.data_augmentations as data_augmentations

# define model, (from scratch or checkpoint)
model = cnn_model(scale_model_width=1) # from scratch
#model = cnn_model.load_from_checkpoint("checkpoints/epoch=999-step=304000.ckpt", scale_model_width=2, map_location=torch.device("cpu")) # from check point

data_path = "./data/datasets/moving_dataset_medium.hdf5"
sound_dir = "./data/reference_data/reference_sounds/"
transform = transforms.Compose([data_augmentations.doppler_aug(max_rel_v=3), data_augmentations.noise_aug(noise_ratio=0.01)])
data_module = MovingImpulseResponseDataModule(data_path,sound_dir,batch_size=10, transform=transform)
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