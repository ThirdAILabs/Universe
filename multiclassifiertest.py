import thirdai
from thirdai import bolt
from thirdai.bolt import MultiLabelTextClassifier

classifier = MultiLabelTextClassifier(input_dim=100000, hidden_layer_dim=1024, n_classes=931)

classifier.train(train_file="wayfair_train.txt", epochs=1, learning_rate=0.001)

classifier.predict(test_file="wayfair_test.txt", output_file="wayfair_out.txt", threshold=0.8)