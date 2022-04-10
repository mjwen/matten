"""
Regression or classification tasks that define the loss function and metrics.

The tasks are helper classes for defining the lighting model.
"""
import abc
from enum import Enum
from typing import Dict

import torch.nn as nn
import torchmetrics
from torch import Tensor
from torchmetrics import (
    AUROC,
    F1,
    Accuracy,
    MeanAbsoluteError,
    MetricCollection,
    Precision,
    Recall,
)


class TaskType(Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    UNKNOWN = "unknown"


class Task:
    """
    Base class for regression or classification task settings.

    Subclass should implement:
        - task_type()
        - init_loss()
        - init_metric()

    Subclass can implement:
        - metric_aggregation()
        - transform_...()

    Args:
        name: name of the task. Values with this key in model prediction dict and
            target dict will be used for loss and metrics computation.
        loss_weight: in multitask learning, e.g. fitting energy an forces together,
            the total loss is a weighted sum of the losses of individual tasks.
            loss_weight gives the weight of this task.
        kwargs: extra information that is needed for the task.
            Kwargs can be either access as an attribute (e.g. instance.<attr_name>) or
            as an item (e.g. instance["<attr_name>"]).
    """

    def __init__(
        self,
        name: str,
        *,
        loss_weight: float = 1.0,
        **kwargs,
    ):
        assert loss_weight > 0, f"`los_weight` should be positive, got {loss_weight}"

        self._name = name
        self._loss_weight = loss_weight

        # store kwargs as attribute
        self.__dict__.update(kwargs)

    @property
    @abc.abstractmethod
    def task_type(self) -> TaskType:
        """
        Type of the task, should be one of TaskType.
        """

    @abc.abstractmethod
    def init_loss(self):
        """
        Initialize the loss for this task.

        Example:
            loss_fn = nn.MSELoss(average='mean')
            return loss_fn
        """

    @abc.abstractmethod
    def init_metric(self) -> torchmetrics.Metric:
        """
        Initialize the metrics (torchmetrics) for the task.

        It could be a plain torchmetric class (e.g. torchmetric.Accuracy) or a
        collection of metrics as in torchmetric.MetricCollection.

        Other programs are not supposed to call this directly, but instead should call
        the wrapper `init_metric_as_collection()`.

        Example 1:
            metric = Accuracy(num_classes=10)
            return metric

        Example 2:
            num_classes = 10
            metric = MetricCollection(
                [
                    Accuracy(num_classes=10),
                    F1(num_classes=10),
                ]
            )
            return metric
        """

    def init_metric_as_collection(self) -> MetricCollection:
        """
        This is a wrapper function for `init_metric()`.

        In `init_metric()`, we allows metric(s) to be any torchmetric object. In this
        function, we convert the metric(s) to a MetriCollection object.
        """
        metric = self.init_metric()
        if not isinstance(metric, MetricCollection):
            metric = MetricCollection([metric])

        return metric

    def metric_aggregation(self) -> Dict[str, float]:
        """
        Ways to aggregate various metrics to a total metric score.

        In the training, some functionality (e.g. early stopping and model checkpoint)
        needs a score to determine its behavior. When multiple metrics are used
        (e.g. using torchmetric.MetricCollection in `init_metric()`), we need to
        determine what metrics contribute to the score and the weight of the metrics
        contribute to the score.

        This function should return a dict: {metric_name: metric_weight}, and it should
        be used together with `init_metric()`.
        `metric_name` is the name of the metric (class name) that contributes to the
        score and `metric_weight` is the corresponding score. The total score is a
        weighted sum of individual scores.

        By default, this is an empty dict, meaning that no total metric score will be
        computed.

        Note:
            Sometimes you may want to use negative weight, depending on the ``mode`` of
            early stopping and mode checkpoing functionality. For example, if we set
            ``mode="min"`` for early stopping and use torchmetric.Accuracy as the metric,
            we need to use a score weight (e.g. -1) to make sure better models
            (higher accuracy) leads to lower score, which is expected by mode="min"
            of the early stopping.


        Example:
            Suppose in `init_metric()`, we have

            metric = MetricCollection(
                [
                    Accuracy(num_classes=10),
                    F1(num_classes=10),
                ]
            )
            return metric

            Then, we can have the below in this function

            metric_agg = {'F1': -1}
            return metric_agg


        Returns:
            {metric_name: metric_weight}, name and weight of a metric
        """

        return {}

    def transform_pred_loss(self, t: Tensor) -> Tensor:
        """
        Transform the model prediction for before provided to loss.

        Note, typically transform_pred_loss, transform_target_loss,
        transform_pred_metric, and transform_target_metric are used together.

        Pseudo code

        input, target = batch
        pred = model(input)

        pred_for_loss = transform_pred_loss(pred)
        target_for_loss = transform_target_loss(target)
        loss_fn(pred_for_loss, target_for_loss)

        pred_for_metric = transform_pred_metric(pred)
        target_for_metric = transform_target_metric(target)
        metric_fn(pred_for_metric, target_for_metric)
        """

        return t

    def transform_target_loss(self, t: Tensor) -> Tensor:
        return t

    def transform_pred_metric(self, t: Tensor) -> Tensor:
        return t

    def transform_target_metric(self, t: Tensor) -> Tensor:
        return t

    def transform_prediction(self, t) -> Tensor:
        """
        Transform the model prediction before returning to user.

        This is supposed to be called in model.forward().
        """
        return t

    @property
    def name(self):
        return self._name

    @property
    def loss_weight(self):
        return self._loss_weight

    def __getitem__(self, key):
        """
        Gets the data of the attribute :obj:`key`.
        """
        return getattr(self, key, None)

    def __setitem__(self, key, value):
        """
        Sets the data of the attribute :obj:`key`.
        """
        setattr(self, key, value)


class ClassificationTask(Task):
    """
    Classification task.

    Subclass should implement:
        - init_loss()
        - init_metric()

    Subclass can implement:
        - is_binary()
        - metric_aggregation()
        - transform()
    """

    def __init__(
        self,
        name: str,
        num_classes: int,
        average: str = "micro",
        *,
        loss_weight: float = 1.0,
        **kwargs,
    ):
        super().__init__(name, loss_weight=loss_weight, **kwargs)
        self.num_classes = num_classes
        self.average = average

    @property
    def task_type(self):
        return TaskType("classification")

    def is_binary(self) -> bool:
        """
        Whether this is a binary classification task.
        """
        return self.num_classes == 2


class CanonicalClassificationTask(ClassificationTask):
    """
    Canonical Classification task with:
        - CrossEntropy loss function (or BCEWithLogitsLoss for binary case)
        - Accuracy, Precision, Recall and F1 metrics
        - F1 contributes to the total metric score
    """

    def init_loss(self):

        # binary classification
        if self.is_binary():
            loss_fn = nn.BCEWithLogitsLoss()

        # multiclass
        else:
            loss_fn = nn.CrossEntropyLoss()

        return loss_fn

    def init_metric(self):

        n = self.num_classes
        average = self.average

        # binary or micro, num_classes not needed
        if self.is_binary() or average == "micro":
            n = None

        m = [
            Accuracy(num_classes=n, average=average, compute_on_step=False),
            Precision(num_classes=n, average=average, compute_on_step=False),
            Recall(num_classes=n, average=average, compute_on_step=False),
            F1(num_classes=n, average=average, compute_on_step=False),
        ]

        # AUROC expect different preds and targets as the other metrics for
        # multiclass classification, so we only enable binary for it now
        if self.is_binary():
            m.append(AUROC(num_classes=n, average=average, compute_on_step=False))

        metric = MetricCollection(m)

        return metric

    def metric_aggregation(self):
        # This requires `mode` of early stopping and checkpoint to be `min`
        # TODO, check the mode?
        return {"F1": -1.0}


class CanonicalRegressionTask(Task):
    """
    Canonical regression task with:
        - MSELoss loss function
        - MeanAbsoluteError metric
        - MeanAbsoluteError contributes to the total metric score
    """

    @property
    def task_type(self):
        return TaskType("regression")

    def init_loss(self):
        return nn.MSELoss()

    def init_metric(self):
        metric = MeanAbsoluteError(compute_on_step=False)

        return metric

    def metric_aggregation(self):

        # This requires `mode` of early stopping and checkpoint to be `min`
        return {"MeanAbsoluteError": 1.0}


if __name__ == "__main__":
    clfn = TaskType("classification")
    print(clfn)
    assert clfn == TaskType.CLASSIFICATION
