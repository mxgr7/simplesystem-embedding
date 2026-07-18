from transformers import AutoTokenizer


def load_fast_tokenizer(model_name):
    """AutoTokenizer with a fallback for pre-tokenizer.json hub repos.

    transformers v5 dropped slow tokenizers, so BERT-era repos that ship
    only a vocab.txt (deepset/gbert-base, bert-base-german-cased, ...)
    fail to load via AutoTokenizer. Rebuild the fast tokenizer from
    vocab.txt in that case.
    """
    try:
        return AutoTokenizer.from_pretrained(model_name)
    except (ValueError, OSError):
        return _build_bert_fast_tokenizer_from_vocab(model_name)


def _build_bert_fast_tokenizer_from_vocab(model_name):
    import json

    from huggingface_hub import hf_hub_download
    from tokenizers.implementations import BertWordPieceTokenizer
    from transformers import PreTrainedTokenizerFast

    vocab_path = hf_hub_download(model_name, "vocab.txt")

    do_lower_case = False
    try:
        config_path = hf_hub_download(model_name, "tokenizer_config.json")
        with open(config_path) as f:
            do_lower_case = bool(json.load(f).get("do_lower_case", False))
    except Exception:
        pass

    backend = BertWordPieceTokenizer(
        vocab_path,
        lowercase=do_lower_case,
        strip_accents=do_lower_case,
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=backend._tokenizer,
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
        model_max_length=512,
    )
