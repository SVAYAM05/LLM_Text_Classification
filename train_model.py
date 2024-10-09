import torch
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2ForSequenceClassification, AdamW
import pandas as pd

# Custom Dataset class
class IMDbDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        input_ids = torch.tensor(self.dataframe.iloc[idx]["input_ids"])
        labels = torch.tensor(self.dataframe.iloc[idx]["labels"])
        return {"input_ids": input_ids, "labels": labels}

# Load preprocessed data
df = pd.read_csv('processed_imdb_data.csv')

# Split data into training and testing sets
train_df = df.sample(frac=0.8, random_state=42)
test_df = df.drop(train_df.index)

# Create datasets and dataloaders
train_dataset = IMDbDataset(train_df)
test_dataset = IMDbDataset(test_df)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4)

# Load GPT-2 model with classification head
model = GPT2ForSequenceClassification.from_pretrained("gpt2", num_labels=2)
optimizer = AdamW(model.parameters(), lr=1e-5)

# Training loop
def train(model, train_loader):
    model.train()
    for epoch in range(3):  # Train for 3 epochs
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"]
            labels = batch["labels"]
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader)}")

if __name__ == "__main__":
    train(model, train_loader)
    model.save_pretrained("./gpt2_text_classifier")
    print("Model training completed and saved.")
