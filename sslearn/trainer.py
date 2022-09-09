import torch
import numpy as np
import warnings

from tqdm import tqdm
from typing import Optional
from torch.utils.data import DataLoader
from torch.optim import Optimizer

from .models import Model
from .models.pretraining import PretrainModel
from .models.finetuning import FinetuneModel
from .validators import Validator

class Trainer:

    def __init__(self, optim: Optimizer, scheduler = None, validator: Optional[Validator] = None):

        self.optim = optim
        self.scheduler = scheduler
        self.validator = validator

    def train(self, model: Model, train_loader: DataLoader, epochs: int, save_freq: int = 10, valid_freq: int = 10, 
              save_path = "weights", device: str = "cuda"):

        """if validator is not None:
            # assert "valid" in dataloaders.keys(), "A validation dataloader must be provided in the 'dataloaders' dictionary under the key 'valid'."
            valid_dl = dataloaders["valid"]
        
        elif "valid" in dataloaders.keys():
            warnings.warn("A validation dataloader has been provided, but a 'validator' has not, and so validation will not occur.")"""

        model.to(device)
        
        losses, valid_metrics = [], []

        for epoch in range(1, epochs+1):
            print("Epoch:", epoch)
            epoch_losses = []
            for images, labels in tqdm(train_loader):
                
                images = images.to(device)

                if isinstance(model, PretrainModel):
                    loss = model.step(images)
                elif isinstance(model, FinetuneModel):
                    labels = labels.to(device)
                    loss = model.step(images, labels)
                else:
                    raise TypeError(f"Model of type '{type(model)}' is not supported.")

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                if self.scheduler is not None:
                    self.scheduler.step()
                
                model.update()

                epoch_losses.append(loss.item())
            
            if epoch % save_freq == 0:

                print("Saving model...")
                torch.save(model.state_dict(), f"{save_path}/{model.name}_{epoch}.pt")
                print("Model saved.")

            avg_loss = np.mean(epoch_losses[-50:])
            losses.append(avg_loss)
            print(f"Avg. train loss: {avg_loss}")

            if (self.validator is not None) and (epoch % valid_freq == 0):

                print("Validating...")
                metric = self.validator(model)
                valid_metrics.append(metric)
                print(f"Validation {self.validator.metric_str}: {metric}")

        return losses, valid_metrics