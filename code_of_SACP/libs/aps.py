import torch


class APS(object):
    def __call__(self, logits, label=None):
        assert len(logits.shape) <= 2, "The dimension of logits must be less than 2."
        if len(logits.shape) == 1:
            logits = logits.unsqueeze(0)
        probs = torch.softmax(logits, dim=-1)
        if label is None:
            return self._calculate_all_label(probs)
        else:
            return self._calculate_single_label(probs, label)

    def _calculate_all_label(self, probs):
        indices, ordered, cumsum = self._sort_sum(probs)
        U = torch.rand(probs.shape, device=probs.device)
        ordered_scores = cumsum - ordered * U
        _, sorted_indices = torch.sort(indices, descending=False, dim=-1)
        scores = ordered_scores.gather(dim=-1, index=sorted_indices)
        return scores

    def _sort_sum(self, probs):
        ordered, indices = torch.sort(probs, dim=-1, descending=True)
        cumsum = torch.cumsum(ordered, dim=-1)
        return indices, ordered, cumsum

    def _calculate_single_label(self, probs, label):
        indices, ordered, cumsum = self._sort_sum(probs)
        U = torch.rand(indices.shape[0], device=probs.device)
        idx = torch.where(indices == label.view(-1, 1))
        scores_first_rank = U * cumsum[idx]
        idx_minus_one = (idx[0], idx[1] - 1)
        scores_usual = U * ordered[idx] + cumsum[idx_minus_one]
        return torch.where(idx[1] == 0, scores_first_rank, scores_usual)
