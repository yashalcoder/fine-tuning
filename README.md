Sentiment Analysis using BERT (Encoder-Only)

This project implements sentiment classification using a pre-trained BERT encoder-only model fine-tuned on a labeled dataset. It classifies input text into:

Positive

Negative

Neutral

📌 Features

Fine-tuned BERT for sentiment analysis

Supports GPU training (Colab compatible)

Evaluation metrics (accuracy, precision, recall, F1)

Confusion matrix & qualitative results

Ready for deployment in Streamlit / FastAPI

📂 Project Structure
.
├── Sentiment_analysis_pre-training_BERT(Encoder_only).ipynb
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── val.csv
├── models/
│   └── saved_model/
├── README.md
└── evaluation_report.md

📦 Installation
pip install torch transformers sklearn pandas numpy matplotlib


For Colab:

!pip install transformers datasets

▶️ Training the Model

Run the notebook:

Sentiment_analysis_pre-training_BERT(Encoder_only).ipynb


The notebook performs:

Tokenization

DataLoader creation

BERT fine-tuning

Evaluation

Saving model

📊 Evaluation

After training, the script generates:

Accuracy

Precision

Recall

F1-Score

Confusion Matrix

Training Loss Plots

🚀 Inference Example
from transformers import BertTokenizer, BertForSequenceClassification
import torch

model_path = "models/saved_model/"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)

text = "This product is amazing!"
inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits
prediction = torch.argmax(logits).item()

print(["Negative", "Neutral", "Positive"][prediction])

💾 Saving & Loading the Model

Model is saved automatically:

models/saved_model/


Load using:

model = BertForSequenceClassification.from_pretrained("models/saved_model")

📈 Results

Accuracy: ~94%

Strong performance on explicit sentiment

Some confusion with sarcasm and ambiguous text

📜 License

This project is open-source and free to use.
GPT-2 Fine-Tuning for Pseudo-Code to C++ Code Generation

This project fine-tunes a GPT-2 decoder-only model to convert structured pseudo-code instructions into valid C++ code. It is trained on 3000 labeled examples and evaluated using BLEU and CodeBLEU.

🚀 Features

GPT-2 model fine-tuned for code generation

Tokenization + dynamic vocab resizing

Training with memory-optimized settings

Automatic model saving to Google Drive

Built-in evaluation (BLEU + CodeBLEU)

Sample predictions included

📂 Project Structure
.
├── notebook.ipynb
├── data/
│   └── dataset.json / csv
├── models/
│   └── gpt2_cpp_safe_final/
├── README.md
└── evaluation_report.md

🔧 Installation
pip install transformers datasets evaluate sentencepiece sacrebleu


For CodeBLEU:

pip install tree-sitter

▶️ Training

Run the training cell:

trainer.train()


Optimizations included:

batch size = 2

gradient accumulation = 8

max_length = 256

ForCausalLMLoss

Model saved to:

/content/drive/MyDrive/gpt2_cpp_safe_final

📊 Evaluation
Results:
Metric	Score
BLEU	0.1416
CodeBLEU	0.4836

Training loss improved significantly across epochs.

🧪 Testing the Model
from transformers import GPT2LMHeadModel, GPT2Tokenizer

path = "/content/drive/MyDrive/gpt2_cpp_safe_final"
model = GPT2LMHeadModel.from_pretrained(path)
tokenizer = GPT2Tokenizer.from_pretrained(path)

prompt = "read integer n and print n*n"
inputs = tokenizer(prompt, return_tensors="pt")

output = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(output[0]))

📈 Sample Outputs
Pseudo-Code	Model Output
declare integer n	int N;
read two integers	Partial (cin >> x;)
for loop 1 to n	Incorrect (N.size())
🛠 Future Improvements

Train for more epochs

Add more code samples

Use CodeLLaMA or GPT-NeoX

Improve pseudo-code formatting

📜 License

This project is open-source and free to modify.
