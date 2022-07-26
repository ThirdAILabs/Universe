from thirdai.bolt import MultiLabelTextClassifier

classifier = MultiLabelTextClassifier(n_classes=931)

classifier.train(train_file="train_pairgrams_10_bert_tok.svm", epochs=3, learning_rate=0.001)
classifier.train(train_file="train_pairgrams_10_bert_tok.svm", epochs=2, learning_rate=0.0001)

classifier.predict(test_file="dev_pairgrams_10_bert_tok.svm", output_file="wayfair_out_temp.txt", threshold=0.5)


# classifier.train(train_file="wayfair_train.txt", epochs=1, learning_rate=0.001)
# classifier.predict(test_file="wayfair_test.txt", output_file="wayfair_out.txt", threshold=0.5)
# classifier.train(train_file="wayfair_train.txt", epochs=1, learning_rate=0.001)
# classifier.predict(test_file="wayfair_test.txt", output_file="wayfair_out.txt", threshold=0.5)

# classifier.train(train_file="wayfair_train.txt", epochs=1, learning_rate=0.0001)
# classifier.predict(test_file="wayfair_test.txt", output_file="wayfair_out.txt", threshold=0.5)
# classifier.train(train_file="wayfair_train.txt", epochs=1, learning_rate=0.0001)
# classifier.predict(test_file="wayfair_test.txt", output_file="wayfair_out.txt", threshold=0.5)
