import torch
import torch.nn.functional as F

class Queue:

    def __init__(self, size: int, encode_dim: int):

        self.size = size
        self.queue = torch.randn(size, encode_dim)
        self.queue = F.normalize(self.queue, dim=-1)
        self.pointer = 0
        #self.queue_tensor = torch.zeros(0, encode_dim)

        """
        torch.zeros(size, encode_dim) and filling in tensor
        should be faster than using torch.cat in enqueue
        """

    """def sample(self, count):
        
        curr_size = len(self.queue)
        rand_idxs = torch.randint(0, curr_size, (count,))
        return self.queue[rand_idxs]"""

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

    """def enqueue(self, batch, max_count=float("inf")):

        batch_size = len(batch)
        end_pointer = self.pointer + batch_size

        if end_pointer > self.size:

            self.queue_tensor[self.pointer : self.size] = batch[:self.size - self.pointer]"""

    """curr_size = len(self.queue_tensor)
        batch = batch[:min(max_count, self.size - curr_size)]
        self.queue_tensor = torch.cat([self.queue_tensor, batch], dim=0)"""
        # check this works if batch self.size - curr_size is == 0

    #def dequeue(self, count):

        #self.queue_tensor = self.queue_tensor[count:]