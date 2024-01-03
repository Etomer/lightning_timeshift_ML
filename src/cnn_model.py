import torch
import torch.nn as nn
import lightning as L
from src.model_pieces import Block

class cnn_model(L.LightningModule):

    def __init__(self,
         n_shift_bins: int = 500,
         dropout: float = 0.1,
         learning_rate: float = 3e-4,
         scale_model_width: int = 1,
         n_frequency_components_input: int = 2500,
         loss_fn = nn.CrossEntropyLoss()
         ):
        super().__init__()
        self.learning_rate = learning_rate
        self.loss_fn  = loss_fn
        
        self.width_at_scale_1 = 576 # Could compute this value with formula depending on self.cnn bellow. Easier just to test 

        self.thinker = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(scale_model_width*self.width_at_scale_1,1000),
            nn.GELU(),
            Block(1000,dropout),
            Block(1000,dropout),
            nn.Linear(1000,n_shift_bins)
        )
        
        self.apply(self._init_weights)
        
        self.cnn = nn.Sequential(
            nn.Conv1d(4,48*scale_model_width, 50,stride=5),
            nn.GELU(),
            nn.Conv1d(48*scale_model_width,48*scale_model_width, 50,stride=5),
            nn.GELU(),
            nn.Conv1d(48*scale_model_width,48*scale_model_width, 30,stride=5),
            nn.GELU(),
            nn.Flatten(),
        )
        

    def forward(self, x):
        x /= x.std(dim=(1,2),keepdim=True) + 1e-5
        x = self.cnn(x)
        x = self.thinker(x)
        return x

    def training_step(self, batch):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log('train/loss', loss, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("val/loss_epoch", loss, on_step=False, on_epoch=True)
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer


    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.0002)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.0002)
        
    