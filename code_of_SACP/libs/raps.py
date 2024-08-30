import torch

from .aps import APS


class RAPS(APS):
    def __init__(self, penalty=0.001, kreg=1):
        super(RAPS, self).__init__()
        self.__penalty = penalty
        self.__kreg = kreg

    def _calculate_all_label(self, probs):
        indices, ordered, cumsum = self._sort_sum(probs)
        U = torch.rand(probs.shape, device=probs.device)
        reg = torch.maximum(self.__penalty * (torch.arange(1, probs.shape[-1] + 1, device=probs.device) - self.__kreg),
                            torch.tensor(0, device=probs.device))
        ordered_scores = cumsum - ordered * U + reg
        _, sorted_indices = torch.sort(indices, descending=False, dim=-1)
        scores = ordered_scores.gather(dim=-1, index=sorted_indices)
        return scores

    def _calculate_single_label(self, probs, label):
        indices, ordered, cumsum = self._sort_sum(probs)
        U = torch.rand(indices.shape[0], device=probs.device)
        idx = torch.where(indices == label.view(-1, 1))
        reg = torch.maximum(self.__penalty * (idx[1] + 1 - self.__kreg), torch.tensor(0).to(probs.device))
        scores_first_rank = U * ordered[idx] + reg
        idx_minus_one = (idx[0], idx[1] - 1)
        scores_usual = U * ordered[idx] + cumsum[idx_minus_one] + reg
        return torch.where(idx[1] == 0, scores_first_rank, scores_usual)
