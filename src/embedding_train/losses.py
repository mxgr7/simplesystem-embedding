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
    return_stats=False,
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
        loss = similarities.sum() * 0.0
        if return_stats:
            return loss, _empty_triplet_stats()
        return loss

    positive_scores = similarities.diagonal()
    same_query_negative_scores, same_query_has_semi_hard = _select_pool_negative_scores(
        similarities, same_query_negative_mask, positive_scores, negative_selection
    )
    cross_query_negative_scores, cross_query_has_semi_hard = _select_pool_negative_scores(
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
    loss = losses.mean()

    if not return_stats:
        return loss

    selected_pool_has_semi_hard = torch.where(
        has_same_query_negative,
        same_query_has_semi_hard,
        cross_query_has_semi_hard,
    )
    valid_anchor_count = int(valid_rows.sum().item())
    semi_hard_fallback_count = int(
        ((~selected_pool_has_semi_hard) & valid_rows).sum().item()
    )
    semi_hard_fallback_share = (
        semi_hard_fallback_count / valid_anchor_count if valid_anchor_count else 0.0
    )

    return loss, {
        "valid_anchor_count": valid_anchor_count,
        "semi_hard_fallback_count": semi_hard_fallback_count,
        "semi_hard_fallback_share": semi_hard_fallback_share,
    }


def _empty_triplet_stats():
    return {
        "valid_anchor_count": 0,
        "semi_hard_fallback_count": 0,
        "semi_hard_fallback_share": 0.0,
    }


def _select_pool_negative_scores(
    similarities, pool_mask, positive_scores, negative_selection
):
    hardest_pool_scores = similarities.masked_fill(~pool_mask, float("-inf")).max(
        dim=1
    ).values

    if negative_selection == "hardest":
        # In hardest mode the "semi-hard found" indicator is meaningless, so we
        # report all anchors as having a semi-hard candidate to keep the
        # fallback-share metric a no-op for this mode.
        return hardest_pool_scores, torch.ones_like(positive_scores, dtype=torch.bool)

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

    selected_scores = torch.where(
        has_semi_hard, hardest_semi_hard_scores, hardest_pool_scores
    )
    return selected_scores, has_semi_hard


def _query_group_ids(query_ids, device):
    query_index_by_id = {}
    query_group_ids = []

    for query_id in query_ids:
        if query_id not in query_index_by_id:
            query_index_by_id[query_id] = len(query_index_by_id)
        query_group_ids.append(query_index_by_id[query_id])

    return torch.tensor(query_group_ids, device=device)
