from src.text_preprocessor.text_vectorizer import TextVectorizer
from src.text_preprocessor.embedding import TextEmbedder


def main():
    text_file = "data/huggingface/train.txt"

    # 1. Build vocabulary
    vectorizer = TextVectorizer(
        max_vocab_size=2_000_000,
        min_freq=1,
        max_len=128,
    )
    vectorizer.fit(text_file)
    print("Vocab size:", len(vectorizer.word2idx))

    # 2. Create embedder
    vocab_size = len(vectorizer.word2idx)
    pad_index = vectorizer.word2idx[vectorizer.PAD_TOKEN]

    embedder = TextEmbedder(
        vocab_size=vocab_size,
        embedding_dim=256,
        pad_index=pad_index,
    )

    # 3. Iterate over file in batches and embed
    batch_size = 1024
    total_sentences = 0

    for batch_num, token_batch in enumerate(
        vectorizer.batch_encode(text_file, batch_size=batch_size)
    ):
        sent_emb = embedder.embed_sentences(token_batch)  # (batch, embedding_dim)
        sent_emb_np = sent_emb.numpy()

        total_sentences += sent_emb_np.shape[0]

        print(
            f"Batch {batch_num} | "
            f"token batch shape = {token_batch.shape} | "
            f"sentence embedding shape = {sent_emb_np.shape}"
        )

        # Optional: save to disk
        # import os, numpy as np
        # os.makedirs("artifacts/embeddings", exist_ok=True)
        # np.save(f"artifacts/embeddings/emb_{batch_num}.npy", sent_emb_np)

        # For debugging you can stop early:
        # if batch_num == 2:
        #     break

    print(f"Total embedded sentences: {total_sentences}")


if __name__ == "__main__":
    main()