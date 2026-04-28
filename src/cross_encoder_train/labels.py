from embedding_train.metrics import RELEVANCE_GAINS

LABEL_ORDER = ("Irrelevant", "Complement", "Substitute", "Exact")
LABEL_TO_ID = {label: index for index, label in enumerate(LABEL_ORDER)}
ID_TO_LABEL = {index: label for label, index in LABEL_TO_ID.items()}
GAIN_VECTOR = tuple(RELEVANCE_GAINS[label] for label in LABEL_ORDER)
NUM_CLASSES = len(LABEL_ORDER)


def encode_label(raw_label):
    if raw_label not in LABEL_TO_ID:
        raise ValueError(f"Unknown relevance label: {raw_label!r}")
    return LABEL_TO_ID[raw_label]
