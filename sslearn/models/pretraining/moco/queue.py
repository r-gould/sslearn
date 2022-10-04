import torch
import torch.nn.functional as F

class Queue:

    def __init__(self, size: int, head_dim: int):

        self.size = size
        self.queue = self._init_queue(size, head_dim)
        self.pointer = 0

    def enqueue_dequeue(self, batch: torch.Tensor):
        
        batch_size = len(batch)
        end_pointer = self.pointer + batch_size

        if end_pointer <= self.size:
            self.queue[self.pointer:end_pointer] = batch
        else:
            end_pointer -= self.size
            diff = self.size - self.pointer
            self.queue[self.pointer:self.size] = batch[:diff]
            self.queue[:end_pointer] = batch[diff:]

        self.pointer = end_pointer

    @staticmethod
    def _init_queue(size, head_dim):

        queue = torch.randn(size, head_dim)
        return F.normalize(queue, dim=-1)