import torch
from .loss import localization_loss, confidence_loss, get_pos_idxs


def localization_loss_metric(preds: torch.Tensor, targs: torch.Tensor) -> float:
    """Return the localization loss of an entire batch (cf the
    localization_loss function, which only operates on a single example)"""
    loc_loss = 0.0
    batch_size = preds.size(0)
    for pred, targ in zip(preds, targs):
        loc_loss += localization_loss(pred, targ) / batch_size

    return loc_loss


def confidence_loss_metric(preds: torch.Tensor, targs: torch.Tensor) -> float:
    """Return the confidence loss of an entire batch (cf the confidence_loss
    function, which only operates on a single example)"""
    conf_loss = 0.0

    batch_size = preds.size(0)
    for pred, targ in zip(preds, targs):
        if len(get_pos_idxs(targ)) == 0:
            continue
        conf_loss += confidence_loss(pred, targ) / batch_size

    return conf_loss


def recall(preds: torch.Tensor, targs: torch.Tensor) -> float:
    """Return the accuracy of default box class predictions"""
    preds = preds[:, :, :-4]
    targs = targs[:, :, :-4]

    batch_size, num_boxes, num_classes = targs.shape
    preds = preds.reshape(batch_size * num_boxes, num_classes)
    targs = targs.reshape(batch_size * num_boxes, num_classes)

    positive_ground_truth_idxs = targs.argmax(dim=1).nonzero().flatten()

    preds = preds[positive_ground_truth_idxs].argmax(dim=1)
    targs = targs[positive_ground_truth_idxs].argmax(dim=1)
    
    recall = (preds == targs).float().mean()
    
    return recall


def precision(preds: torch.Tensor, targs: torch.Tensor) -> float:
    """Return the accuracy of default box class predictions"""
    preds = preds[:, :, :-4]
    targs = targs[:, :, :-4]

    batch_size, num_boxes, num_classes = targs.shape
    preds = preds.reshape(batch_size * num_boxes, num_classes)
    targs = targs.reshape(batch_size * num_boxes, num_classes)

    positive_pred_idxs = preds.argmax(dim=1).nonzero().flatten()
    
    preds = preds[positive_pred_idxs].argmax(dim=1)
    targs = targs[positive_pred_idxs].argmax(dim=1)
    
    precision = (preds == targs).float().mean()
    
    return precision
