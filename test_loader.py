import time
from thirdai.dataset import blocks
from thirdai.dataset import text_encodings
from thirdai.dataset import loaders
from thirdai.dataset import StreamingDataset

def wayfair():
    # Define source
    csv_loader = loaders.CSV("/Users/benitogeordie/Desktop/thirdai_datasets/train_auto_classifier.csv")

    # Label is categorical
    label = blocks.Categorical(col=0, dim=931)

    # Unigrams for first 4 words
    unigram_0 = blocks.Text(1, text_encodings.UniGram(dim=50_000, start_pos=0, end_pos=1))
    unigram_1 = blocks.Text(1, text_encodings.UniGram(dim=50_000, start_pos=1, end_pos=2))
    unigram_2 = blocks.Text(1, text_encodings.UniGram(dim=50_000, start_pos=2, end_pos=3))
    unigram_3 = blocks.Text(1, text_encodings.UniGram(dim=50_000, start_pos=3, end_pos=4))
    # Pairgrams for all words
    pairgrams = blocks.Text(1, text_encodings.PairGram(dim=500_000))

    # Assemble
    dataset = StreamingDataset(
        loader=csv_loader, 
        input_blocks=[unigram_0, unigram_1, unigram_2, unigram_3, pairgrams], 
        target_blocks=[label], 
        batch_size=2048,
        est_num_samples=32000000)

    start = time.time()
    dataset.load_in_memory()
    end = time.time()
    print("Loaded Wayfair data in", end - start, "seconds.")

def amazon_polarity():
    filename = "/Users/benitogeordie/Desktop/thirdai_datasets/amazon_polarity/amazon_polarity_train.txt"
    
    # Assemble
    dataset = StreamingDataset(
        loader=loaders.CSV(filename, delimiter="\t"), 
        input_blocks=[blocks.Categorical(col=0, dim=2)], 
        target_blocks=[blocks.Text(1)], 
        batch_size=2048,
        est_num_samples=3600000,
        # )
        shuffle=True,
        shuffle_buffer_size=20480)

    start = time.time()
    dataset.load_in_memory()
    end = time.time()
    print("Loaded Amazon Polarity data in", end - start, "seconds.")

amazon_polarity()