import os, sys, torch
sys.path.append(os.getcwd())
from lightning.pytorch.loggers import WandbLogger
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from torchvision import transforms

# choose model and data_module --------------------------------
from src.data_modules import MovingImpulseResponseDataModule
from src.cnn_model_v2 import cnn_model_v2
import src.data_augmentations as data_augmentations

L.seed_everything(42, workers=True)

# define model, (from scratch or checkpoint)
model = cnn_model_v2(scale_cnn_width=2, n_blocks=4, block_width=2000) # from scratch

#model = cnn_model_v2.load_from_checkpoint("lightning_logs/haslkohp/checkpoints/epoch=2999-step=39000.ckpt", scale_model_width=2, n_blocks=4, block_width=2000) # from check point

#data_path = "./data/datasets/moving_dataset_medium.hdf5"
data_path = "data/datasets/moving_dataset_directivity_medium_extra_val2.hdf5"
data_val_path = "data/datasets/moving_dataset_directivity_medium_extra_val.hdf5"
sound_dir = "./data/reference_data/reference_sounds/"
transform = transforms.Compose([data_augmentations.doppler_aug(max_rel_v=1), data_augmentations.noise_aug(noise_ratio=0.1)])
data_module = MovingImpulseResponseDataModule(data_path,sound_dir,batch_size=20, transform=transform, n_mics_per_batch=17, data_val_path=data_val_path)
#-----------------------------------------------
wandb_logger = WandbLogger(log_model="all")
checkpoint_callback = ModelCheckpoint(every_n_epochs=50)
trainer = L.Trainer(
    max_epochs=30000,
    accelerator="cuda",
    devices=[0],#[1],
    logger=wandb_logger,
    log_every_n_steps=50,
    #terminate_on_nan=True,
    callbacks=[checkpoint_callback],
    deterministic=True,
)
trainer.fit(model, data_module)