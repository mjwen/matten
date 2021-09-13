"""
Base Lightning model for regression and classification.
"""
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import pytorch_lightning as pl
import torch
import torch.nn as nn
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from torch import Tensor
from torch.optim import lr_scheduler

from eigenn.model.task import ClassificationTask, RegressionTask, Task
from eigenn.model.utils import TimeMeter


class BaseModel(pl.LightningModule):
    """

    Subclass must implement:
        - init_backbone(): create the underlying torch model
        - init_tasks(): create tasks that defines initialize the loss function and metrics

    subclass may implement:
        - decode(): compute model prediction using the torch model
        - compute_loss(): compute the loss using model prediction and the target
    """

    def __init__(self, **kwargs):
        super().__init__()

        self.save_hyperparameters()

        # backbone model
        self.backbone = self.init_backbone(self.hparams)

        # tasks
        tasks = self.init_tasks(self.hparams)
        if isinstance(tasks, Task):
            tasks = [tasks]
        assert isinstance(tasks, Sequence)
        # set tasks as a dict, with task name as key and task object as value
        self.tasks = {t.name: t for t in tasks}

        # losses
        self.loss_fns = {name: task.init_loss() for name, task in self.tasks.items()}

        # metrics
        # dict of dict: {mode: {task_name: metric_object}}
        self.metrics = nn.ModuleDict()
        for mode in ["train", "val", "test"]:
            # cannot use `train` directly (already a submodule of the class)
            mode = "metric_" + mode
            self.metrics[mode] = nn.ModuleDict()
            for name, task in self.tasks.items():
                mc = task.init_metric_as_collection()
                self.metrics[mode][name] = mc

        # timer
        self.timer = TimeMeter()

    def init_backbone(self, params) -> nn.Module:
        """
        Create a backbone torch model.

        A pytorch or lightning model that can be called like:
        `model(graphs, feats, metadata)`
        The model should return a dictionary of {task_name: task_prediction}, where the
        task_name should the name of one task defined in `init_tasks()` and
        task_prediction should be a tensor.

        This will be called in the `decode()` function.
        Oftentimes, the underlying model may not return a dictionary (e.g. when using
        existing models). In this case, the model prediction should be converted to a
        dictionary in the `decode()` function.
        """
        raise NotImplementedError

    def init_tasks(self, params) -> Dict:
        """
        Define the tasks used to compute loss and metrics.

        This should return a `Task` instance of a list of `Task` instances.

        Example:
            from eigenn.model.task import ClassificationTask
            t = ClassificationTask(name='crystal_type', num_classes=10)
            return t
        """

        raise NotImplementedError

    def forward(
        self,
        graphs,
        feats: [str, Any] = None,
        metadata: Dict[str, Any] = None,
        mode: Optional[str] = None,
    ):
        """
        Args:
            graphs:
            feats:
            metadata:
            mode: select what to return. See below.

        Returns:
            If `None`, directly return the value returned by the backbone forward method.
        """

        if mode is None:
            return self.backbone(graphs, feats, metadata)
        elif mode == "decoder":
            return self.decode(graphs, feats, metadata)
        else:
            raise ValueError(f"Not supported return mode: {mode}")

    def decode(
        self,
        graphs,
        feats: Dict[str, Tensor],
        metadata: Dict[str, Tensor],
    ) -> Dict[str, Tensor]:
        """
        Compute prediction for each task using the backbone model.

        Args:
            graphs: batched graphs
            feats: batched features for the model
            metadata: extra metadata needed by the model to compute the prediction

        Returns:
            {task_name: task_prediction}
        """

        preds = self.backbone(graphs, feats, metadata)

        return preds

    def compute_loss(
        self, preds: Dict[str, Tensor], labels: Dict[str, Tensor]
    ) -> Tuple[Dict[str, Tensor], Tensor]:
        """
        Compute the loss for each task.

        Args:
            preds: {task_name, prediction} prediction for each task
            labels: {task_name, label} labels for each task

        Returns:
            individual_loss: {task_name: loss} loss of individual task
            total_loss: total loss, weighted sum of individual loss
        """
        individual_losses = {}
        total_loss = 0.0

        for task_name, task in self.tasks.items():
            p = preds[task_name]
            l = labels[task_name]
            if task.task_type() == "classification" and task.is_binary():
                # this will use BCEWithLogitsLoss, which requires label be of float
                p = p.reshape(-1)
                l = l.reshape(-1).to(torch.get_default_dtype())

            loss_fn = self.loss_fns[task_name]
            loss = loss_fn(p, l)
            individual_losses[task_name] = loss
            total_loss = total_loss + task.loss_weight * loss

        return individual_losses, total_loss

    def training_step(self, batch, batch_idx):
        loss, preds, labels = self.shared_step(batch, "train")
        self.update_metrics(preds, labels, "train")

        return {"loss": loss}

    def training_epoch_end(self, outputs):
        self.compute_metrics("train")

    def validation_step(self, batch, batch_idx):
        loss, preds, labels = self.shared_step(batch, "val")
        self.update_metrics(preds, labels, "val")

        # TODO, remove this
        # return {"loss": loss}

    def validation_epoch_end(self, outputs):
        _, score = self.compute_metrics("val")

        # val/score used for early stopping and learning rate scheduler
        if score is not None:
            self.log(f"val/score", score, on_step=False, on_epoch=True, prog_bar=True)

        # time it
        delta_t, cumulative_t = self.timer.update()
        self.log("epoch time", delta_t, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "cumulative time", cumulative_t, on_step=False, on_epoch=True, prog_bar=True
        )

    def test_step(self, batch, batch_idx):
        loss, preds, labels = self.shared_step(batch, "test")
        self.update_metrics(preds, labels, "test")

        # TODO, remove this
        # return {"loss": loss}

    def test_epoch_end(self, outputs):
        self.compute_metrics("test")

    def shared_step(self, batch, mode: str):
        """
        Shared computation step.

        Args:
            batch: data batch, obtained from dataloader
            mode: train, val, or test
        """

        # ========== compute predictions ==========
        graphs, labels, metadata = batch

        # lightning cannot move graphs to gpu, so do it manually
        graphs = graphs.to(self.device)

        # TODO, get feats from batched graphs
        # nodes = ["atom", "global"]
        # feats = {nt: mol_graphs.nodes[nt].data.pop("feat") for nt in nodes}
        # feats["bond"] = mol_graphs.edges["bond"].data.pop("feat")
        feats = None

        preds = self.decode(graphs, feats, metadata)

        # ========== compute losses ==========
        individual_loss, total_loss = self.compute_loss(preds, labels)

        self.log_dict(
            {
                f"{mode}/loss/{task_name}": loss
                for task_name, loss in individual_loss.items()
            },
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )

        self.log(
            f"{mode}/total_loss",
            total_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        return total_loss, preds, labels

    def update_metrics(self, preds, labels, mode):
        """
        Update metric values at each step, i.e. keep record of values of each step that
        will be used to compute the epoch metrics.

        Args:
            mode: train, val, or test
        """
        mode = "metric_" + mode

        for task_name, metric in self.metrics[mode].items():

            task = self.tasks[task_name]

            # regression metrics
            if isinstance(task, RegressionTask):
                pr, la = task.transform(preds[task_name], labels[task_name])
                metric(pr, la)

            # classification metrics
            elif isinstance(task, ClassificationTask):
                p = preds[task_name]

                # binary
                if p.shape[1] == 1:
                    prob = torch.sigmoid(p.reshape(-1))

                # multiclass
                else:
                    # prop = torch.softmax(p, dim=1)
                    # torch.softmax maybe unstable (e.g. sum of the values is away from 1
                    # due to numerical errors), which torchmetrics will complain
                    prob = torch.argmax(p, dim=1)

                metric(prob, labels[task_name])

            else:
                raise RuntimeError(f"Unsupported task type {task.__class__}")

    def compute_metrics(
        self, mode, log: bool = True
    ) -> Tuple[Dict[str, Tensor], Union[Tensor, None]]:
        """
        Compute metric and logger it at each epoch.

        Args:
            log: whether to log the metrics

        Returns:
            individual_score: individual metric scores, {task_name: scores},
                where scores is a dict.
            score: aggregated score. `None` if metric_aggregation() of task is not set.
        """

        mode = "metric_" + mode

        total_score = None
        individual_score = {}

        for task_name, metric_coll in self.metrics[mode].items():

            # metric collection output, a dict: {metric_name: metric_value}
            score = metric_coll.compute()
            individual_score[task_name] = score

            if log:
                for metric_name, metric_value in score.items():
                    self.log(
                        f"{mode}/{metric_name}/{task_name}",
                        metric_value,
                        on_step=False,
                        on_epoch=True,
                        prog_bar=False,
                    )

            # compute score for model checkpoint and early stopping
            task = self.tasks[task_name]
            metric_agg_dict = task.metric_aggregation()
            if metric_agg_dict:
                total_score = 0 if total_score is None else total_score
                for metric_name, weight in metric_agg_dict.items():
                    total_score = total_score + score[metric_name] * weight

            # reset to initial state for next epoch
            metric_coll.reset()

        return individual_score, total_score

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        # learning rate scheduler
        scheduler = self._config_lr_scheduler(optimizer)

        if scheduler is None:
            return optimizer
        else:
            return {
                "optimizer": optimizer,
                "lr_scheduler": scheduler,
                "monitor": "val/score",
            }

    def _config_lr_scheduler(self, optimizer):

        scheduler_name = self.hparams.lr_scheduler["scheduler_name"].lower()

        if scheduler_name == "reduce_on_plateau":
            scheduler = lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", factor=0.4, patience=50, verbose=True
            )
        elif scheduler_name == "cosine":
            scheduler = LinearWarmupCosineAnnealingLR(
                optimizer,
                warmup_epochs=self.hparams.lr_scheduler["lr_warmup_step"],
                max_epochs=self.hparams.lr_scheduler["epochs"],
                eta_min=self.hparams.lr_scheduler["lr_min"],
            )
        elif scheduler_name == "none":
            scheduler = None
        else:
            raise ValueError(f"Not supported lr scheduler: {self.hparams.lr_scheduler}")

        return scheduler
