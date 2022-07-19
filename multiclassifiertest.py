from thirdai.bolt import MultiLabelTextClassifier

classifier = MultiLabelTextClassifier(n_classes=931)

classifier.train(train_file="wayfair_train.txt", epochs=3, learning_rate=0.001)
classifier.train(train_file="wayfair_train.txt", epochs=2, learning_rate=0.0001)

classifier.predict(test_file="wayfair_test.txt", output_file="wayfair_out_with_top.txt", threshold=0.5)

