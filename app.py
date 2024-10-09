from flask import Flask, request, jsonify
import torch
from transformers import GPT2ForSequenceClassification, GPT2Tokenizer

# Initialize Flask app
app = Flask(__name__)

# Load the trained model and tokenizer
model = GPT2ForSequenceClassification.from_pretrained("./gpt2_text_classifier")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data['text']
    input_ids = tokenizer.encode(text, return_tensors="pt", max_length=512, truncation=True)
    with torch.no_grad():
        outputs = model(input_ids=input_ids)
        prediction = torch.argmax(outputs.logits, dim=-1).item()
    return jsonify({'sentiment': 'positive' if prediction == 1 else 'negative'})

if __name__ == '__main__':
    app.run(debug=True)
