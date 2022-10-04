import torch
import numpy as np
import warnings

from tqdm import tqdm
from typing import Optional
from torch.utils.data import DataLoader
from torch.optim import Optimizer

from ..models import _Model
from ..models.pretraining import _PretrainModel
from ..models.finetuning import _FinetuneModel
from .validators import _Validator

class Trainer:

    def __init__(
        self,
        optim: Optimizer,
        scheduler = None,
        validator: Optional[_Validator] = None
    ):
        self.optim = optim
        self.scheduler = scheduler
        self.validator = validator

    def train(
        self,
        model: _Model,
        train_loader: DataLoader,
        epochs: int,
        save_path: str,
        save_freq: int = 10,
        valid_freq: int = 10, 
        #save_path = "weights",
        device: str = "cuda"
    ):
        model.to(device)
        
        losses, valid_metrics = [], []

        for epoch in range(1, epochs+1):
            print("Epoch:", epoch)
            epoch_losses = []
            for images, labels in tqdm(train_loader):
                
                images = images.to(device)

                if isinstance(model, _PretrainModel):
                    loss = model.step(images)
                elif isinstance(model, _FinetuneModel):
                    labels = labels.to(device)
                    loss = model.step(images, labels)
                else:
                    raise TypeError(f"Model of type '{type(model)}' is not supported.")

                print("LOSS:", loss.item())
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