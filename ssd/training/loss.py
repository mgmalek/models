import torch
import torch.nn as nn


smooth_l1 = nn.SmoothL1Loss(reduction='sum')
cross_entropy_loss = nn.CrossEntropyLoss(reduction='none')


def get_pos_idxs(targ: torch.Tensor) -> torch.Tensor:
    """Get the indices of the default boxes in the target tensor that
    correspond to positive examples (i.e. default boxes that have been matched
    with annotations)"""
    pos_idxs = targ[:, :-4].argmax(dim=1).nonzero(as_tuple=False).flatten()
    return pos_idxs


def localization_loss(pred: torch.Tensor, targ: torch.Tensor) -> torch.Tensor:
    """Calculate localisation loss, which measures the performance of
    bounding box offset predictions"""
    pos_idxs = get_pos_idxs(targ)

    pred_offsets = pred[pos_idxs, -4:]
    targ_offsets = targ[pos_idxs, -4:]

    loc_loss = smooth_l1(pred_offsets.flatten(), targ_offsets.flatten())

    return loc_loss


def confidence_loss(pred: torch.Tensor, targ: torch.Tensor, neg_pos_ratio: int = 3) -> torch.Tensor:
    """Calculate confidence loss, which measures the performance of
    bounding box class predictions

    To calculate confidence loss, use all positive examples plus
    the negative examples with the highest confidence loss such
    that the ratio between positives and negatives is at most 3:1"""

    pos_idxs = get_pos_idxs(targ)
    num_examples = len(targ)
    neg_idxs = (torch.arange(num_examples).view(-1, 1).cuda() != pos_idxs).min(dim=1).values.nonzero().flatten()
    max_neg_examples = len(pos_idxs) * neg_pos_ratio

    # Calculate confidence loss of positive examples
    pred_pos = pred[pos_idxs, :-4]
    targ_pos = targ[pos_idxs, :-4].argmax(dim=1)

    pos_conf_losses = cross_entropy_loss(pred_pos, targ_pos)

    # Calculate confidence loss of a limited number of negative examples
    pred_neg = pred[neg_idxs, :-4]
    targ_neg = targ[neg_idxs, :-4].argmax(dim=1)

    neg_conf_losses = cross_entropy_loss(pred_neg, targ_neg)
    highest_neg_conf_loss_idxs = neg_conf_losses.argsort(descending=True)
    highest_neg_conf_loss_idxs = highest_neg_conf_loss_idxs[:max_neg_examples]

    highest_neg_conf_losses = neg_conf_losses[highest_neg_conf_loss_idxs]

    # Calculate overall confidence loss
    conf_loss = pos_conf_losses.sum() + highest_neg_conf_losses.sum()
    return conf_loss


def loss_func(preds: torch.Tensor, targs: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """Loss function for SSD training. Loss is a linear combination of
    confidence loss (for class prediction) and localization loss (for offset
    prediction)"""
    loss = 0.0
    for pred, targ in zip(preds, targs):
        pos_idxs = get_pos_idxs(targ)
        num_matched_boxes = len(pos_idxs)

        if num_matched_boxes == 0:
            continue

        loc_loss = localization_loss(pred, targ)
        conf_loss = confidence_loss(pred, targ)
        loss += (conf_loss + alpha * loc_loss) / num_matched_boxes

    batch_size = preds.size(0)
    loss /= batch_size
    
    return loss
