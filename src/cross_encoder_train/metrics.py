from cross_encoder_train.labels import ID_TO_LABEL, NUM_CLASSES


def compute_classification_metrics(prediction_ids, target_ids):
    if len(prediction_ids) != len(target_ids):
        raise ValueError(
            "prediction_ids and target_ids must have the same length: "
            f"{len(prediction_ids)} vs {len(target_ids)}"
        )

    confusion = [[0 for _ in range(NUM_CLASSES)] for _ in range(NUM_CLASSES)]
    for predicted, target in zip(prediction_ids, target_ids):
        confusion[int(target)][int(predicted)] += 1

    total = sum(sum(row) for row in confusion)
    correct = sum(confusion[i][i] for i in range(NUM_CLASSES))

    metrics = {
        "accuracy": (correct / total) if total else 0.0,
        "evaluated_pairs": float(total),
    }

    f1_values = []
    for class_index in range(NUM_CLASSES):
        true_positive = confusion[class_index][class_index]
        false_positive = (
            sum(confusion[i][class_index] for i in range(NUM_CLASSES))
            - true_positive
        )
        false_negative = sum(confusion[class_index]) - true_positive
        support = sum(confusion[class_index])

        precision = (
            true_positive / (true_positive + false_positive)
            if (true_positive + false_positive)
            else 0.0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if (true_positive + false_negative)
            else 0.0
        )
        f1 = (
            (2 * precision * recall) / (precision + recall)
            if (precision + recall)
            else 0.0
        )
        f1_values.append(f1)

        suffix = ID_TO_LABEL[class_index].lower()
        metrics[f"precision_{suffix}"] = precision
        metrics[f"recall_{suffix}"] = recall
        metrics[f"f1_{suffix}"] = f1
        metrics[f"support_{suffix}"] = float(support)

    metrics["macro_f1"] = sum(f1_values) / NUM_CLASSES
    return metrics
