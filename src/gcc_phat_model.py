import torch
import torch.nn as nn
import lightning as L
from src.model_pieces import Block

class gcc_phat_model(L.LightningModule):

    def __init__(self,
         n_shift_bins: int ,
         max_shift : int, 
         sample_length : int,
         n_frequency_components_input: int = 2500,
         loss_fn = nn.CrossEntropyLoss(),
         temperature_scaling : float = 1,
         ):
        super().__init__()
        self.sample_length = sample_length
        self.max_shift = max_shift
        self.n_frequency_components_input = n_frequency_components_input
        self.n_shift_bins = n_shift_bins
        self.loss_fn  = loss_fn
        self.temperature_scaling = temperature_scaling
        
    def forward(self, x):
        x = torch.complex(x[:, 0::2, :], x[:, 1::2])
        c = x[:, 0] * x[:, 1].conj()
        x = torch.fft.irfft(
            torch.concatenate(
                [
                    c / (c.abs() + 1e-10),
                    torch.zeros(
                        x.shape[0], 1 + self.sample_length // 2 - x.shape[2]
                    ),
                ],
                dim=1,
            )
        )

        x = torch.fft.fftshift(x, dim=1)
        # recompute  correct bin size
        bin_size = 2 * self.max_shift/ self.n_shift_bins
        pred = torch.zeros(x.shape[0], self.n_shift_bins)
        for i in range(self.n_shift_bins):
            start = int(
                x.shape[1] // 2
                - (self.n_shift_bins / 2) * bin_size
                + bin_size * i
            )
            # print(torch.sum(x[:,start:(start + bin_size)],dim=1).shape)
            pred[:, i] = torch.sum(x[:, start : (start + int(bin_size))], dim=1)
        pred = pred*self.temperature_scaling
        return pred

    def training_step(self, batch):
        raise Exception("GCC-phat-model should not be trained!")
        
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("val/loss_epoch", loss, on_step=False, on_epoch=True)
        

    def configure_optimizers(self):
        raise Exception("GCC-phat-model should not be trained!")