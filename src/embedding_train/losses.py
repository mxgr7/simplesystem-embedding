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
    query_embeddings,
    offer_embeddings,
    query_ids,
    labels,
    margin,
    negative_selection="semi_hard",
):
    if negative_selection not in {"semi_hard", "hardest"}:
        raise ValueError(
            f"Unsupported negative_selection: {negative_selection}. "
            "Expected 'semi_hard' or 'hardest'."
        )

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
    same_query_negative_scores = _select_pool_negative_scores(
        similarities, same_query_negative_mask, positive_scores, negative_selection
    )
    cross_query_negative_scores = _select_pool_negative_scores(
        similarities, cross_query_negative_mask, positive_scores, negative_selection
    )
    selected_negative_scores = torch.where(
        has_same_query_negative,
        same_query_negative_scores,
        cross_query_negative_scores,
    )
    losses = F.relu(
        selected_negative_scores[valid_rows] - positive_scores[valid_rows] + margin
    )
    return losses.mean()


def _select_pool_negative_scores(
    similarities, pool_mask, positive_scores, negative_selection
):
    hardest_pool_scores = similarities.masked_fill(~pool_mask, float("-inf")).max(
        dim=1
    ).values

    if negative_selection == "hardest":
        return hardest_pool_scores

    # Semi-hard: prefer the hardest negative whose similarity is still below
    # the positive's. Negatives already above the positive dominate gradients
    # and cause early collapse, so they are masked out here. When no negative
    # is below the positive for an anchor we fall back to the hardest in the
    # pool so the triplet still contributes a loss.
    below_positive_mask = pool_mask & (similarities < positive_scores.unsqueeze(1))
    hardest_semi_hard_scores = similarities.masked_fill(
        ~below_positive_mask, float("-inf")
    ).max(dim=1).values
    has_semi_hard = below_positive_mask.any(dim=1)

    return torch.where(has_semi_hard, hardest_semi_hard_scores, hardest_pool_scores)


def _query_group_ids(query_ids, device):
    query_index_by_id = {}
    query_group_ids = []

    for query_id in query_ids:
        if query_id not in query_index_by_id:
            query_index_by_id[query_id] = len(query_index_by_id)
        query_group_ids.append(query_index_by_id[query_id])

    return torch.tensor(query_group_ids, device=device)
