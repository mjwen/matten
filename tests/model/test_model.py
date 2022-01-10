import torch
import torch.nn as nn
from pytorch_lightning import seed_everything

from eigenn.model.model import BaseModel
from eigenn.model.task import CanonicalClassificationTask

seed_everything(seed=35)

TASK_NAME = "my_task"


class Backbone(nn.Module):
    def __init__(self, input_dim=4):
        super().__init__()
        self.layer = nn.Linear(in_features=input_dim, out_features=1)

    def forward(self, graphs):
        output = self.layer(graphs)
        return {TASK_NAME: output}


class BasicModel(BaseModel):
    def init_backbone(self, hparams, dataset_hparams):
        backbone = Backbone()
        return backbone

    def init_tasks(self, tasks):
        # binary classification
        task = CanonicalClassificationTask(name=TASK_NAME, num_classes=2)

        return {TASK_NAME: task}

    def decode(self, model_input):
        return self.backbone(model_input)


def test_basic_model():
    # data
    input = torch.FloatTensor([[0, 0, 1, 2], [1, 0, 1, 2]])
    labels = {TASK_NAME: torch.tensor([0, 1])}

    # model
    model = BasicModel()
    preds = model.decode(input)

    # output
    reference = torch.tensor([[-0.4218], [-0.6598]])
    assert torch.allclose(preds[TASK_NAME], reference, rtol=1e-4)

    # loss
    individual_loss, total_loss = model.compute_loss(preds, labels)
    assert torch.allclose(total_loss, torch.tensor(0.7904), rtol=1e-4)

    # test various methods
    model.update_metrics(preds, labels, mode="train")
    individual_score, total_score = model.compute_metrics(mode="train", log=False)

    ref_score = {
        "Accuracy": 0.5,
        "Precision": 0.0,
        "Recall": 0.0,
        "F1": 0.0,
        "AUROC": 0.0,
    }
    for k, v in individual_score[TASK_NAME].items():
        assert v == ref_score[k]
    assert total_score == 0
