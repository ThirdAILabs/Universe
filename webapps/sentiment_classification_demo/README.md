# Product Search Demo

This demo allows for the user to input sentences and see both the sentiment and latency as classified by bolt and RoBERTa.

## Steps to Run App

1. Download the hugging face amazon polarity dataset.
2. Preprocess the dataset using the `preprocess_amazon_polarity.py` script in `/webapps/sentiment_classification_demo/`.
3. Train the bolt model and save it. 
4. Run `python3 webapps/sentiment_classification_demo/app.py <path to saved model>`. Flask will report the ip address and port the app is running on.