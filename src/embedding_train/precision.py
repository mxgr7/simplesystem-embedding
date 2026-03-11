import torch


SUPPORTED_EMBEDDING_PRECISIONS = {
    "float32",
    "float16",
    "int8",
    "sign",
    "binary",
}

POPCOUNT_LOOKUP = torch.tensor(
    [bin(value).count("1") for value in range(256)],
    dtype=torch.uint8,
)


def resolve_embedding_precision(precision):
    normalized = str(precision).strip().lower()

    if normalized in SUPPORTED_EMBEDDING_PRECISIONS:
        return normalized

    choices = "|".join(sorted(SUPPORTED_EMBEDDING_PRECISIONS))
    raise ValueError(
        f"Unsupported embedding precision: {precision}. Expected one of {choices}"
    )


def quantize_embeddings(embeddings, precision):
    resolved_precision = resolve_embedding_precision(precision)
    embeddings = embeddings.detach().cpu()

    if resolved_precision == "float32":
        return embeddings.to(dtype=torch.float32)

    if resolved_precision == "float16":
        return embeddings.to(dtype=torch.float16)

    if resolved_precision == "int8":
        scaled = torch.round(embeddings.to(dtype=torch.float32) * 127.0)
        return scaled.clamp(min=-127, max=127).to(dtype=torch.int8)

    if resolved_precision == "sign":
        return torch.where(
            embeddings >= 0,
            torch.ones_like(embeddings, dtype=torch.int8),
            -torch.ones_like(embeddings, dtype=torch.int8),
        )

    return pack_binary_embeddings(embeddings)


def pack_binary_embeddings(embeddings):
    bit_tensor = (embeddings.detach().cpu() >= 0).to(dtype=torch.uint8)
    batch_size, dimension = bit_tensor.shape
    padded_dimension = ((dimension + 7) // 8) * 8

    if padded_dimension != dimension:
        padding = torch.zeros(
            (batch_size, padded_dimension - dimension),
            dtype=torch.uint8,
        )
        bit_tensor = torch.cat([bit_tensor, padding], dim=1)

    weights = torch.tensor([128, 64, 32, 16, 8, 4, 2, 1], dtype=torch.uint8)
    packed = bit_tensor.reshape(batch_size, -1, 8)
    return (packed * weights).sum(dim=2).to(dtype=torch.uint8)


def serialize_embeddings(embeddings, precision):
    resolved_precision = resolve_embedding_precision(precision)

    if resolved_precision == "binary":
        return [bytes(row.tolist()) for row in embeddings]

    return embeddings.tolist()


def score_embedding_pairs(query_embeddings, offer_embeddings, precision):
    resolved_precision = resolve_embedding_precision(precision)

    if resolved_precision in {"float32", "float16"}:
        query_values = quantize_embeddings(query_embeddings, resolved_precision).to(
            dtype=torch.float32
        )
        offer_values = quantize_embeddings(offer_embeddings, resolved_precision).to(
            dtype=torch.float32
        )
        return (query_values * offer_values).sum(dim=1)

    if resolved_precision == "int8":
        query_values = quantize_embeddings(query_embeddings, resolved_precision).to(
            dtype=torch.int32
        )
        offer_values = quantize_embeddings(offer_embeddings, resolved_precision).to(
            dtype=torch.int32
        )
        scores = (query_values * offer_values).sum(dim=1).to(dtype=torch.float32)
        return scores / (127.0 * 127.0)

    if resolved_precision == "sign":
        query_values = quantize_embeddings(query_embeddings, resolved_precision).to(
            dtype=torch.int16
        )
        offer_values = quantize_embeddings(offer_embeddings, resolved_precision).to(
            dtype=torch.int16
        )
        scores = (query_values * offer_values).sum(dim=1).to(dtype=torch.float32)
        return scores / float(query_values.size(1))

    query_values = quantize_embeddings(query_embeddings, resolved_precision)
    offer_values = quantize_embeddings(offer_embeddings, resolved_precision)
    xor_values = torch.bitwise_xor(query_values, offer_values)
    differing_bits = POPCOUNT_LOOKUP[xor_values.to(dtype=torch.long)].sum(dim=1)
    dimension = query_embeddings.size(1)
    scores = dimension - (2.0 * differing_bits.to(dtype=torch.float32))
    return scores / float(dimension)
