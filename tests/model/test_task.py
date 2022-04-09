import torch

from eigenn.model.task import CanonicalClassificationTask, CanonicalRegressionTask


def test_classification_task():
    name = "some_name"
    loss_weight = 2.0
    num_classes = 3

    task = CanonicalClassificationTask(
        name=name, num_classes=num_classes, loss_weight=loss_weight
    )

    # property
    assert task.name == name
    assert task.loss_weight == loss_weight

    # metrics
    metric = task.init_metric_as_collection()
    preds = torch.tensor([0, 0, 1, 2])
    labels = torch.tensor([1, 0, 1, 2])
    metric(preds, labels)
    out = metric.compute()
    for name in metric.keys():
        assert out[name] == 0.75


def test_regression_task():
    name = "some_test"
    loss_weight = 2.0

    task = CanonicalRegressionTask(name=name, loss_weight=loss_weight)

    # property
    assert task.name == name
    assert task.loss_weight == loss_weight

    # metrics
    metric = task.init_metric_as_collection()
    preds = torch.FloatTensor([0, 0, 1, 2])
    labels = torch.FloatTensor([1, 0, 1, 2])
    metric(preds, labels)
    out = metric.compute()
    assert out["MeanAbsoluteError"] == 0.25

    # transform
    preds_transform = task.transform_pred_loss(preds)
    labels_transform = task.transform_target_metric(labels)

    assert torch.allclose(preds_transform, preds)
    assert torch.allclose(labels_transform, labels)
