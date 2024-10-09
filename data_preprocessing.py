import pandas as pd
from sklearn.datasets import fetch_openml
from transformers import GPT2Tokenizer

# Fetch the IMDb dataset
def load_data():
    imdb_data = fetch_openml("imdb_reviews", version=1)
    df = pd.DataFrame(imdb_data.data, columns=["review", "sentiment"])
    return df

# Tokenize the text and encode labels
def preprocess_data(df):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    df["input_ids"] = df["review"].apply(lambda x: tokenizer.encode(x, max_length=512, truncation=True))
    df["labels"] = df["sentiment"].apply(lambda x: 1 if x == "pos" else 0)
    return df

if __name__ == "__main__":
    df = load_data()
    df = preprocess_data(df)
    df.to_csv('processed_imdb_data.csv', index=False)  # Save preprocessed data
    print("Data Preprocessed and saved to 'processed_imdb_data.csv'.")
