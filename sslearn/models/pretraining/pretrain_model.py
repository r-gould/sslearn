from ..model import Model

class PretrainModel(Model):

    name = "pretrain"

    def step(self, x):

        raise NotImplementedError()