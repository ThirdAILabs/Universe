with open("/Users/benitogeordie/Desktop/thirdai_datasets/amazon_polarity_train_preprocessed.txt", "w") as f:
    for line in open("/Users/benitogeordie/Desktop/thirdai_datasets/amazon_polarity_train.txt"):
        f.write(line[:1])
        f.write('\t')
        f.write(line[2:])