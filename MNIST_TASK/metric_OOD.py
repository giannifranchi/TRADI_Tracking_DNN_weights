import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
import anom_utils
import torch
import torch.nn as nn




def eval_ood_measure(conf, seg_label, pred, out_labels=[10], mask=None):
    ECE = _ECELoss()
    ece_out = ECE.forward(conf,pred,seg_label)
    conf=conf.cpu().numpy()
    seg_label = seg_label.cpu().numpy()
    if mask is not None:
        seg_label = seg_label[mask]

    out_label = seg_label == out_labels[0]

    for label in out_labels:
        out_label = np.logical_or(out_label, seg_label == label)

    in_scores = - conf[np.logical_not(out_label)]
    out_scores  = - conf[out_label]

    if (len(out_scores) != 0) and (len(in_scores) != 0):
        auroc, aupr, fpr = anom_utils.get_and_print_results(out_scores, in_scores)
        return auroc, aupr, fpr, ece_out.cpu().item()
    else:
        print("This image does not contain any OOD pixels or is only OOD.",out_labels )
        return None


class _ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).
    The input to this loss is the logits of a model, NOT the softmax scores.
    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:
    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |
    We then return a weighted average of the gaps, based on the number
    of samples in each bin
    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """
    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(_ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, confidences, predictions, labels):

        accuracies = predictions.eq(labels)
        ece = torch.zeros(1, device=confidences.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece