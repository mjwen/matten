"""
Regression or classification tasks that define the loss function and metrics.
"""

from typing import Dict, Optional, Tuple

import torch.nn as nn
from torch import Tensor
from torchmetrics import (
    F1,
    Accuracy,
    MeanAbsoluteError,
    MetricCollection,
    Precision,
    Recall,
)


class Task:
    """
    Base class for regression or classification task settings.

    Subclass should implement:
        - init_loss()
        - init_metric()

    Subclass can implement:
        - score_metric()
        - transform()

    Args:
        name: name of the task
        loss_weight: in multi-task learning, e.g. fitting energy an forces together,
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

        self.name = name
        self.loss_weight = loss_weight

        # store kwargs as attribute
        self.__dict__.update(kwargs)

    def init_loss(self):
        """
        Initialize the loss for this task.

        Example:
            loss_fn = nn.MSELoss(average='mean')
            return loss_fn
        """
        raise NotImplementedError

    def init_metric(self):
        """
        Initialize the metrics (torchmetrics) for the task.

        It could be a plain torchmetric class (e.g. torchmetric.Accuracy) or a
        collection of metrics as in torchmetric.MetricCollection.

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
        raise NotImplementedError

    def score_metric(
        self,
    ) -> Dict[str, str]:
        """
        Name of the metric to compute a validation score and the weight for the score.

        In the training, some functionality (e.g. early stopping and model checkpoint)
        needs a score to determine its behavior. When multiple metrics are used
        (e.g. using torchmetric.MetricCollection in `init_metric()`), we need to
        determine what metrics are used as the score and the weight it contributes to
        the score.

        This function should return a dict: {metric_name: metric_weight}, and it should
        be used together with `init_metric()`.
        `metric_name` is the name of the metric that contributes to the score and
        `metric_weight` is the corresponding score. The total score is a weighted sum
        of individual scores.

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

            score_metric = {'F1': -1}
            return score_metric


        Returns:
            {metric_name: metric_weight}, name and weight of a metric
        """

        return {}

    def transform(self, preds: Tensor, labels: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Transform the task predictions and labels.

        This is typically used for regression task, not classification task.
        For regression task, we may scale the target by subtracting the mean and
        then dividing the standard deviation. To recover the prediction in the
        original space, we should reverse the process. Inverse transformation like this
        can be implemented here.

        Args:
            preds: model prediction
            labels: reference labels for the prediction

        Returns:
            transformed_preds, transformed_labels
        """
        return preds, labels

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
    Canonical Classification task with CrossEntropy loss with Accuracy, Precision,
    Recall and F1 metrics.
    """

    def __init__(
        self,
        name: str,
        num_classes: int,
        average: str = "micro",
        *,
        loss_weight: float = 1.0,
        score_metric_name: str = None,
        score_weight: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(
            name,
            loss_weight=loss_weight,
            score_metric_name=score_metric_name,
            score_weight=score_weight,
            **kwargs,
        )
        self.num_classes = num_classes
        self.average = average

    def init_loss(self):
        loss_fn = nn.CrossEntropyLoss()
        return loss_fn

    def init_metric(self):

        n = self.num_classes
        average = self.average

        # binary or micro, num_classes not needed
        if n == 2 or average == "micro":
            n = None

        metric = MetricCollection(
            [
                Accuracy(num_classes=n, average=average, compute_on_step=False),
                Precision(num_classes=n, average=average, compute_on_step=False),
                Recall(num_classes=n, average=average, compute_on_step=False),
                F1(num_classes=n, average=average, compute_on_step=False),
            ]
        )

        return metric


class RegressionTask(Task):
    def init_loss(self):
        return nn.MSELoss()

    def init_metric(self):
        metric = MeanAbsoluteError(compute_on_step=False)

        return metric
