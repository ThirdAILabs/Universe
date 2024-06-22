from utils import run_and_time
import tqdm
import numpy as np

def main(num_cpus):
    print("Downloading dataset...")
    from datasets import load_dataset
    ds = load_dataset("mteb/amazon_polarity", "default")

    print("Downloading model...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("avsolatorio/GIST-small-Embedding-v0")
    
    print("Preparing multi process pool...")
    pool, _ = run_and_time(lambda: model.start_multi_process_pool(['cpu'] * num_cpus))

    def embed_with_progress(data):
        batch_size = 10_000
        batches = []
        for i in tqdm.tqdm(range(0, len(data), batch_size)):
            batches.append(model.encode_multi_process(data[i:i+batch_size], pool, batch_size=256))
        return np.concatenate(batches)

    print("Embedding train text...")
    train_embeddings, _ = run_and_time(lambda: embed_with_progress(ds["train"]["text"]))
    print("Embedding val text...")
    val_embeddings, _ = run_and_time(lambda: embed_with_progress(ds["validation"]["text"]))
    print("Embedding test text...")
    test_embeddings, _ = run_and_time(lambda: embed_with_progress(ds["test"]["text"]))

    print("Closing pool...")
    model.stop_multi_process_pool(pool)

    print("Saving embeddings...")
    train_embeddings.dump("train_embeddings.np")
    val_embeddings.dump("val_embeddings.np")
    test_embeddings.dump("test_embeddings.np")


if __name__ == "__main__":
    import sys
    main(num_cpus=int(sys.argv[1]) if len(sys.argv) > 1 else 2)