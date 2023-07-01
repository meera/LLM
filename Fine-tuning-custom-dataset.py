# Install the latest version of the transformers library
# !pip install transformers -U

# Import libraries
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from transformers import TrainingArguments, Trainer
from transformers import BertTokenizer, BertForSequenceClassification

# Visit this link to download the dataset.  I downloaded the train.csv file.
# https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/data
data = pd.read_csv("C:\Users\15209\__YouTube_Summarizor\train\train.csv")


# Select only the comment_text and toxic columns. Then select only the first 1000 rows of data.
data = data[['comment_text','toxic']]
data = data[0:1000] 


# Load tokenizer and model.  You can add any model, but I am using the bert-base-uncased model.
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased',num_labels=2)


# Split data into training and validation sets using sklearn. Then tokenize the data.
X = list(data["comment_text"])
y = list(data["toxic"])
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2,stratify=y)
X_train_tokenized = tokenizer(X_train, padding=True, truncation=True, max_length=512)
X_val_tokenized = tokenizer(X_val, padding=True, truncation=True, max_length=512)


# Create torch dataset.  This is a custom dataset class that we will use to pass the tokenized data to the model.
class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels:
            item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])
    

# Create torch datasets by calling the Dataset class above
train_dataset = Dataset(X_train_tokenized, y_train)
val_dataset = Dataset(X_val_tokenized, y_val)


# Define the compute_metrics function. This function will be passed to the Trainer class below.
def compute_metrics(p):
    print(type(p))
    pred, labels = p
    pred = np.argmax(pred, axis=1)

    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    recall = recall_score(y_true=labels, y_pred=pred)
    precision = precision_score(y_true=labels, y_pred=pred)
    f1 = f1_score(y_true=labels, y_pred=pred)

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


# Define the trainer arguments.  This is where you can define the number of epochs, batch size, etc.
args = TrainingArguments(
    output_dir="output",
    num_train_epochs=1,
    per_device_train_batch_size=8

)
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)


# This will start to train the model
trainer.train()

# Evaluate the model
trainer.evaluate()

# Save the model
trainer.save_model("model")