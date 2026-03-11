import faiss


SUPPORTED_FAISS_INDEX_TYPES = {
    "flat",
    "ivf_flat",
    "ivf_pq",
    "hnsw",
}


def resolve_faiss_index_type(index_type):
    normalized = str(index_type).strip().lower()

    if normalized in SUPPORTED_FAISS_INDEX_TYPES:
        return normalized

    choices = "|".join(sorted(SUPPORTED_FAISS_INDEX_TYPES))
    raise ValueError(
        f"Unsupported FAISS index type: {index_type}. Expected one of {choices}"
    )


def index_requires_training(index_type):
    return resolve_faiss_index_type(index_type) in {"ivf_flat", "ivf_pq"}


def validate_faiss_index_args(index_type, embedding_dim, index_config):
    resolved_index_type = resolve_faiss_index_type(index_type)

    if resolved_index_type in {"ivf_flat", "ivf_pq"} and int(index_config["nlist"]) < 1:
        raise ValueError("--nlist must be at least 1")

    if resolved_index_type == "ivf_pq":
        pq_m = int(index_config["pq_m"])
        pq_bits = int(index_config["pq_bits"])

        if pq_m < 1:
            raise ValueError("--pq-m must be at least 1")

        if pq_bits < 1:
            raise ValueError("--pq-bits must be at least 1")

        if int(embedding_dim) % pq_m != 0:
            raise ValueError(
                f"Embedding dimension {embedding_dim} must be divisible by pq_m={pq_m}"
            )

    if resolved_index_type == "hnsw":
        if int(index_config["hnsw_m"]) < 2:
            raise ValueError("--hnsw-m must be at least 2")

        if int(index_config["ef_construction"]) < 2:
            raise ValueError("--ef-construction must be at least 2")

        if int(index_config["ef_search"]) < 1:
            raise ValueError("--ef-search must be at least 1")


def build_index_config(args):
    return {
        "index_type": resolve_faiss_index_type(args.index_type),
        "nlist": int(args.nlist),
        "train_sample_size": int(args.train_sample_size),
        "nprobe": int(args.nprobe),
        "pq_m": int(args.pq_m),
        "pq_bits": int(args.pq_bits),
        "hnsw_m": int(args.hnsw_m),
        "ef_construction": int(args.ef_construction),
        "ef_search": int(args.ef_search),
    }


def minimum_training_vectors(index_type, index_config):
    resolved_index_type = resolve_faiss_index_type(index_type)

    if resolved_index_type == "ivf_flat":
        return max(1, int(index_config["nlist"]))

    if resolved_index_type == "ivf_pq":
        return max(
            int(index_config["nlist"]),
            1 << int(index_config["pq_bits"]),
        )

    return 0


def create_faiss_index(embedding_dim, index_config):
    resolved_index_type = resolve_faiss_index_type(index_config["index_type"])
    validate_faiss_index_args(resolved_index_type, embedding_dim, index_config)

    if resolved_index_type == "flat":
        return faiss.IndexIDMap2(faiss.IndexFlatIP(embedding_dim))

    if resolved_index_type == "ivf_flat":
        quantizer = faiss.IndexFlatIP(embedding_dim)
        return faiss.IndexIVFFlat(
            quantizer,
            embedding_dim,
            int(index_config["nlist"]),
            faiss.METRIC_INNER_PRODUCT,
        )

    if resolved_index_type == "ivf_pq":
        quantizer = faiss.IndexFlatIP(embedding_dim)
        return faiss.IndexIVFPQ(
            quantizer,
            embedding_dim,
            int(index_config["nlist"]),
            int(index_config["pq_m"]),
            int(index_config["pq_bits"]),
            faiss.METRIC_INNER_PRODUCT,
        )

    base_index = faiss.IndexHNSWFlat(
        embedding_dim,
        int(index_config["hnsw_m"]),
        faiss.METRIC_INNER_PRODUCT,
    )
    base_index.hnsw.efConstruction = int(index_config["ef_construction"])
    base_index.hnsw.efSearch = int(index_config["ef_search"])
    return faiss.IndexIDMap2(base_index)


def unwrap_faiss_index(index):
    if hasattr(index, "index"):
        try:
            return faiss.downcast_index(index.index)
        except Exception:
            pass

    return faiss.downcast_index(index)


def apply_search_parameters(index, index_config, nprobe=None, ef_search=None):
    resolved_index_type = resolve_faiss_index_type(index_config["index_type"])

    if resolved_index_type in {"ivf_flat", "ivf_pq"}:
        configured_nprobe = nprobe
        if configured_nprobe is None:
            configured_nprobe = index_config.get("nprobe")

        if configured_nprobe is not None:
            index.nprobe = int(configured_nprobe)
        return

    if resolved_index_type == "hnsw":
        configured_ef_search = ef_search
        if configured_ef_search is None:
            configured_ef_search = index_config.get("ef_search")

        if configured_ef_search is not None:
            unwrap_faiss_index(index).hnsw.efSearch = int(configured_ef_search)
