import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

german_train = pd.read_csv("/content/train_v1.4.tsv", sep="\t", header=None)
german_test = pd.read_csv("/content/set_v1.4.tsv", sep="\t", header=None)

german_train.columns = ["id", "text", "relevance", "sentiment", "aspect"]
german_test.columns = ["id", "text", "relevance", "sentiment", "aspect"]

print(german_train.head())
print(german_train.columns)

german_train = german_train[["text", "sentiment"]]
german_test = german_test[["text", "sentiment"]]

german_train.dropna(inplace=True)
german_test.dropna(inplace=True)

print(german_train["sentiment"].value_counts())

turkish_data = pd.read_csv("/content/turkish_test.csv")

print(turkish_data.head())
print(turkish_data.columns)

print(turkish_data["label"].value_counts())

turkish_data["label"] = turkish_data["label"].replace({
    "Positive": "positive",
    "Negative": "negative",
    "Notr": "neutral"
})

turkish_data = turkish_data[["text", "label"]]

turkish_data.columns = ["text", "sentiment"]

print(turkish_data.head())
print(turkish_data.columns)

print(turkish_data["sentiment"].value_counts())

X_train_ger = german_train["text"]
y_train_ger = german_train["sentiment"]

X_test_ger = german_test["text"]
y_test_ger = german_test["sentiment"]

X_test_tur = turkish_data["text"]
y_test_tur = turkish_data["sentiment"]

vectorizer = TfidfVectorizer(max_features=5000)

X_train_ger_tfidf = vectorizer.fit_transform(X_train_ger)

X_test_ger_tfidf = vectorizer.transform(X_test_ger)

X_test_tur_tfidf = vectorizer.transform(X_test_tur)

baseline_model = LogisticRegression(max_iter=1000)

baseline_model.fit(X_train_ger_tfidf,y_train_ger)

y_pred_ger = baseline_model.predict(X_test_ger_tfidf)

accuracy_ger = accuracy_score(y_test_ger,y_pred_ger)

print("German to German Accuracy:", accuracy_ger)

print(classification_report(y_test_ger,y_pred_ger))

print(confusion_matrix(y_test_ger,y_pred_ger))

y_pred_tur = baseline_model.predict(X_test_tur_tfidf)

accuracy_tur = accuracy_score(y_test_tur,y_pred_tur)

print("German to Turkish Accuracy:", accuracy_tur)

print(classification_report(y_test_tur,y_pred_tur))

print(confusion_matrix(y_test_tur,y_pred_tur))

ConfusionMatrixDisplay.from_predictions(y_test_ger, y_pred_ger, cmap="Blues")

plt.title("German to German Confusion Matrix")
plt.show()

ConfusionMatrixDisplay.from_predictions(y_test_tur, y_pred_tur, cmap="Oranges")

plt.title("German to Turkish Confusion Matrix")

plt.show()

"""### **BASELINE 2 IMPLEMENTATION**"""

!pip install transformers datasets sentencepiece accelerate -q

import torch
from datasets import Dataset
from transformers import (XLMRobertaTokenizer, XLMRobertaForSequenceClassification, Trainer, TrainingArguments)

label_mapping = {"negative": 0, "neutral": 1, "positive": 2}

y_train_encoded = [label_mapping[label] for label in y_train_ger]

y_test_encoded = [label_mapping[label] for label in y_test_ger]

train_df = pd.DataFrame({"text": X_train_ger, "label": y_train_encoded})

test_df = pd.DataFrame({"text": X_test_ger, "label": y_test_encoded})

train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")

def tokenize_function(example):
  return tokenizer(
        example["text"],
        padding="max_length",
        truncation=True,
        max_length=128)

train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

model = XLMRobertaForSequenceClassification.from_pretrained("xlm-roberta-base", num_labels=3)

training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
    weight_decay=0.01,
    logging_steps=50,
    save_strategy="no"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

trainer.train()

german_predictions = trainer.predict(test_dataset)

german_pred_labels = np.argmax(german_predictions.predictions, axis=1)

accuracy_ger_xlmr = accuracy_score(y_test_encoded, german_pred_labels)

print("German-to-German XLM-R Accuracy:")
print(accuracy_ger_xlmr)

print(classification_report(y_test_encoded, german_pred_labels))

ConfusionMatrixDisplay.from_predictions(y_test_encoded, german_pred_labels, cmap="Blues")

plt.title( "German-to-German XLM-R Confusion Matrix")
plt.show()

turkish_sample = turkish_data.sample(n=200, random_state=42).reset_index(drop=True)

y_true_xlmr = [label_mapping[label] for label in turkish_sample["sentiment"]]

turkish_dataset = Dataset.from_pandas(pd.DataFrame({"text": turkish_sample["text"]}))

turkish_dataset = turkish_dataset.map(tokenize_function, batched=True)

turkish_predictions = trainer.predict(turkish_dataset)

turkish_pred_labels = np.argmax(turkish_predictions.predictions, axis=1)

xlmr_accuracy = accuracy_score(y_true_xlmr, turkish_pred_labels)

print("German-to-Turkish XLM-R Accuracy:")
print(xlmr_accuracy)

print(classification_report(y_true_xlmr, turkish_pred_labels))

print(confusion_matrix(y_true_xlmr, turkish_pred_labels))

ConfusionMatrixDisplay.from_predictions(y_true_xlmr, turkish_pred_labels, cmap="Purples")

plt.title("German-to-Turkish XLM-R Confusion Matrix")
plt.show()