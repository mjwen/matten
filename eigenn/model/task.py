"""
Regression or classification tasks that define the loss function and metrics.

The tasks are helper classes for defining the lighting model.
"""

from typing import Dict, Optional, Tuple

import numpy as np
import numpy.linalg as linalg
import scipy.linalg
import torch
import torch.nn as nn
import torchmetrics
from scipy.linalg import logm, sqrtm
from torch import Tensor
from torchmetrics import (
    AUROC,
    F1,
    Accuracy,
    MeanAbsoluteError,
    MeanSquaredError,
    MetricCollection,
    Precision,
    Recall,
)


class Task:
    """
    Base class for regression or classification task settings.

    Subclass should implement:
        - task_type()
        - init_loss()
        - init_metric()

    Subclass can implement:
        - metric_aggregation()
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

        self._name = name
        self._loss_weight = loss_weight
        self._task_type = None

        # store kwargs as attribute
        self.__dict__.update(kwargs)

    def task_type(self) -> str:
        """
        Type of the task, should be either `classification` or `regression`.

        Returns:
            type of the task, either `classification` or `regression`.
        """
        raise NotImplementedError

    def init_loss(self):
        """
        Initialize the loss for this task.

        Example:
            loss_fn = nn.MSELoss(average='mean')
            return loss_fn
        """
        raise NotImplementedError

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
        raise NotImplementedError

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

    def task_type(self):
        return "classification"

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


class RegressionTask(Task):
    """
    Regression task.

    Subclass should implement:
        - init_loss()
        - init_metric()

    Subclass can implement:
        - metric_aggregation()
        - transform()

    Args:
         label_transform_dict: information to transform the label to its original space
            using the mean and standard deviation. Currently, this should be of
            {'mean': Tensor, 'std': Tensor}. This is optional: if
            `None`, no transform is performed.
    """

    def __init__(
        self,
        name: str,
        *,
        loss_weight: float = 1.0,
        label_transform_dict: Optional[Dict[str, Tensor]] = None,
        **kwargs,
    ):
        super().__init__(name, loss_weight=loss_weight, **kwargs)
        self.label_transform_dict = self._check_label_transform_dict(
            label_transform_dict
        )

    def task_type(self):
        return "regression"

    def transform(self, preds: Tensor, labels: Tensor) -> Tuple[Tensor, Tensor]:

        if self.label_transform_dict is not None:
            mean = self.label_transform_dict["mean"]
            std = self.label_transform_dict["std"]
            preds = preds * std + mean
            labels = labels * std + mean

        return preds, labels

    @staticmethod
    def _check_label_transform_dict(d):
        if d is not None:
            keys = set(d.keys())
            expected_keys = {"mean", "std"}
            assert keys == expected_keys, (
                f"Expect `label_transform_dict` to be `None` or a dict with keys "
                f"{expected_keys}. Got {d}."
            )

        return d


class CanonicalRegressionTask(RegressionTask):
    """
    Canonical regression task with:
        - MSELoss loss function
        - MeanAbsoluteError metric
        - MeanAbsoluteError contributes to the total metric score
    """

    def init_loss(self):
        return nn.MSELoss()

    def init_metric(self):
        metric = MeanAbsoluteError(compute_on_step=False)

        return metric

    def metric_aggregation(self):

        # This requires `mode` of early stopping and checkpoint to be `min`
        return {"MeanAbsoluteError": 1.0}


def adjoint(A, E, f):
    A_H = A.T.conj().to(E.dtype)
    n = A.size(0)
    M = torch.zeros(2 * n, 2 * n, dtype=E.dtype, device=E.device)
    M[:n, :n] = A_H
    M[n:, n:] = A_H
    M[:n, n:] = E
    return f(M)[:n, n:].to(A.dtype)


def logm_scipy(A):
    return torch.from_numpy(scipy.linalg.logm(A.cpu(), disp=False)[0]).to(A.device)


class Logm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A):
        assert A.ndim == 2 and A.size(0) == A.size(1)  # Square matrix
        assert A.dtype in (
            torch.float32,
            torch.float64,
            torch.complex64,
            torch.complex128,
        )
        ctx.save_for_backward(A)
        return logm_scipy(A)

    @staticmethod
    def backward(ctx, G):
        (A,) = ctx.saved_tensors
        return adjoint(A, G, logm_scipy)


def sqrtm_scipy(A):
    return torch.from_numpy(scipy.linalg.sqrtm(A.cpu(), disp=False)[0]).to(A.device)


class Sqrtm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A):
        assert A.ndim == 2 and A.size(0) == A.size(1)  # Square matrix
        assert A.dtype in (
            torch.float32,
            torch.float64,
            torch.complex64,
            torch.complex128,
        )
        ctx.save_for_backward(A)
        return sqrtm_scipy(A)

    @staticmethod
    def backward(ctx, G):
        (A,) = ctx.saved_tensors
        return adjoint(A, G, sqrtm_scipy)


class GeodesicLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, yhat, y):
        sqrtm = Sqrtm.apply
        loss = torch.tensor(0, dtype=torch.float64)
        for A, B in zip(yhat, y):
            inv_sqrt = linalg.inv(sqrtm(yhat))
            temp = np.matmul(inv_sqrt, np.matmul(y, inv_sqrt))
            eig, vecs = linalg.eig(temp)
            dist_sum = 0
            for i in eig:
                dist_sum += torch.log(i) ** 2
            loss = loss + dist_sum
        return torch.sqrt(dist_sum)


class LogGeodesicLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, yhat, y):
        logm = Logm.apply
        loss = torch.tensor(0, dtype=torch.float64)
        for A, B in zip(yhat, y):
            loss = loss + (torch.sqrt(torch.trace((logm(A) - logm(B)) ** 2)))
        return loss


class MaxRegressionTask(RegressionTask):
    """
    Regression task with:
        - MSELoss loss function
        - MeanAbsoluteError metric
        - MeanAbsoluteError contributes to the total metric score
    """

    def __init__(
        self,
        name: str,
        *,
        loss_fn: str = "l2",
        tensor_shape: str = "symmetric",
        loss_weight: float = 1.0,
        label_transform_dict: Optional[Dict[str, Tensor]] = None,
        **kwargs,
    ):
        super().__init__(name, loss_weight=loss_weight, **kwargs)
        self.label_transform_dict = self._check_label_transform_dict(
            label_transform_dict
        )
        self.loss_fn = loss_fn
        self.tensor_shape = tensor_shape

    def init_loss(self):
        if self.loss_fn == "l2":
            return nn.MSELoss()
        elif self.loss_fn == "l1":
            return nn.L1Loss()
        elif self.loss_fn == "g":
            return GeodesicLoss()
        elif self.loss_fn == "lg":
            return LogGeodesicLoss()

    def init_metric(self):

        m = [
            MeanAbsoluteError(compute_on_step=False),
            MeanSquaredError(compute_on_step=False),
            # TODO This throws error, learn how to implement the loss as a metric
        ]

        metric = MetricCollection(m)

        return metric

    def metric_aggregation(self):

        # This requires `mode` of early stopping and checkpoint to be `min`
        return {"MeanAbsoluteError": 1.0}
