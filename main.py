import tensorflow as tf
from src.utils.config import load_config
from src.data.tokenizer import load_tokenizer
from src.data.tf_dataloader import preprocess_datasets, create_tf_datasets
from src.models.embedding import TokenAndPositionEmbedding
from src.models.masks import create_padding_mask
from src.models.encoder import Encoder


def main():
    # 1) Load config
    config = load_config()

    # 2) Load tokenizer (only for text -> ids)
    tokenizer = load_tokenizer(config)

    # 3) Load + tokenize CNN/DailyMail, then convert to tf.data.Dataset
    train_enc, val_enc, test_enc = preprocess_datasets(config, tokenizer)
    train_tf, val_tf, test_tf = create_tf_datasets(train_enc, val_enc, test_enc, config)

    # 4) Get one batch from the training pipeline
    batch = next(iter(train_tf))
    x_dict, y = batch

    input_ids = x_dict["input_ids"]              # (B, L)
    attention_mask = x_dict["attention_mask"]    # (B, L)

    print("input_ids shape:", input_ids.shape)

    # 5) Build x = token_embedding + positional_encoding
    d_model = config["model"]["d_model"]
    max_len = config["dataset"]["max_input_len"]
    vocab_size = tokenizer.vocab_size
    pad_token_id = tokenizer.pad_token_id

    embed_layer = TokenAndPositionEmbedding(vocab_size, d_model, max_len)
    x = embed_layer(input_ids)                   # (B, L, d_model)

    # 6) Build padding mask for encoder self-attention
    pad_mask = create_padding_mask(input_ids, pad_token_id)  # (B, 1, 1, L)

    print("x shape for encoder:", x.shape)
    print("pad_mask shape:", pad_mask.shape)

    # 7) Run encoder (your custom Transformer encoder)
    num_layers = config["model"]["num_encoder_layers"]
    num_heads = config["model"]["num_heads"]
    d_ff = config["model"]["d_ff"]
    dropout = config["model"].get("dropout", 0.1)

    encoder = Encoder(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        dropout=dropout
    )

    enc_out, attn = encoder(x, mask=pad_mask, training=False)   # (B, L, d_model)

    print("encoder output shape:", enc_out.shape)

    # Optional: inspect attention weights shape from first layer
    first_key = "encoder_layer_1"
    if first_key in attn:
        print("attention weights (layer 1) shape:", attn[first_key].shape)  # (B, H, L, L)


if __name__ == "__main__":
    main()