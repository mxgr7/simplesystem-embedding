import torch
import torch.nn.functional as F


def cosine_bce_loss(scores, labels, scale, positive_weight=1.0, negative_weight=1.0):
    logits = scores * scale
    losses = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    weights = torch.where(
        labels > 0.5,
        torch.full_like(labels, positive_weight),
        torch.full_like(labels, negative_weight),
    )
    return (losses * weights).mean()


def in_batch_contrastive_loss(
    query_embeddings, offer_embeddings, query_ids, labels, scale
):
    query_group_ids = _query_group_ids(query_ids, labels.device)
    logits = torch.matmul(query_embeddings, offer_embeddings.transpose(0, 1)) * scale

    positive_offers = labels > 0.5
    positive_mask = query_group_ids.unsqueeze(1) == query_group_ids.unsqueeze(0)
    positive_mask = positive_mask & positive_offers.unsqueeze(0)
    valid_rows = positive_mask.any(dim=1)

    if not valid_rows.any():
        return logits.sum() * 0.0

    log_probs = F.log_softmax(logits, dim=1)
    positive_log_probs = log_probs.masked_fill(~positive_mask, float("-inf"))
    losses = -torch.logsumexp(positive_log_probs[valid_rows], dim=1)
    return losses.mean()


def in_batch_triplet_loss(
    query_embeddings, offer_embeddings, query_ids, labels, margin
):
    query_group_ids = _query_group_ids(query_ids, labels.device)
    similarities = torch.matmul(query_embeddings, offer_embeddings.transpose(0, 1))

    positive_anchors = labels > 0.5
    same_query = query_group_ids.unsqueeze(1) == query_group_ids.unsqueeze(0)
    positive_offers = positive_anchors.unsqueeze(0)
    same_query_negative_mask = same_query & ~positive_offers
    cross_query_negative_mask = ~same_query
    has_same_query_negative = same_query_negative_mask.any(dim=1)
    has_cross_query_negative = cross_query_negative_mask.any(dim=1)
    valid_rows = positive_anchors & (has_same_query_negative | has_cross_query_negative)

    if not valid_rows.any():
        return similarities.sum() * 0.0

    positive_scores = similarities.diagonal()
    same_query_negative_scores = similarities.masked_fill(
        ~same_query_negative_mask, float("-inf")
    )
    cross_query_negative_scores = similarities.masked_fill(
        ~cross_query_negative_mask, float("-inf")
    )
    hardest_same_query_negative_scores = same_query_negative_scores.max(dim=1).values
    hardest_cross_query_negative_scores = cross_query_negative_scores.max(dim=1).values
    hardest_negative_scores = torch.where(
        has_same_query_negative,
        hardest_same_query_negative_scores,
        hardest_cross_query_negative_scores,
    )
    losses = F.relu(
        hardest_negative_scores[valid_rows] - positive_scores[valid_rows] + margin
    )
    return losses.mean()


def _query_group_ids(query_ids, device):
    query_index_by_id = {}
    query_group_ids = []

    for query_id in query_ids:
        if query_id not in query_index_by_id:
            query_index_by_id[query_id] = len(query_index_by_id)
        query_group_ids.append(query_index_by_id[query_id])

    return torch.tensor(query_group_ids, device=device)
