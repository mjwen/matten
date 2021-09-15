import torch.nn as nn

from eigenn.model.model import BaseModel
from eigenn.model.task import CanonicalRegressionTask

TASK_NAME = "n"


class Backbone(nn.Module):
    def __init__(self, input_dim=4):
        super().__init__()
        self.layer = nn.Linear(in_features=input_dim, out_features=1)

    def forward(self, graphs, labels=None, metadata=None):
        output = self.layer(graphs)
        return {TASK_NAME: output}


class BasicModel(BaseModel):
    def init_backbone(self, hparams):
        backbone = Backbone()
        return backbone

    def init_tasks(self, hparams):
        task = CanonicalRegressionTask(name=TASK_NAME)

        return task
