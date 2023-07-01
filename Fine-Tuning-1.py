# Set up the environment

#!pip install -q transformers datasets
#!pip install transformers[torch]
#!pip install pytorch-trainer
#!pip install accelerate -U
#!pip install trainer
#!pip install --upgrade protobuf


#Load dataset
# From Hugging Face, download a multi-label text classification dataset from the hub. Here 
# I'm using the SemEval 2018 Task 1 dataset, which contains tweets and their associated labels. 
# The labels are multi-label, meaning that each tweet can have multiple labels. 
# For example, the tweet "I love apples and oranges" could have the labels "positive" and "food".
from datasets import load_dataset

dataset = load_dataset("sem_eval_2018_task_1", "subtask5.english")

#  Create a list that contains the labels, as well as 2 dictionaries that map labels to integers and back.
labels = [label for label in dataset['train'].features.keys() if label not in ['ID', 'Tweet']]
id2label = {idx:label for idx, label in enumerate(labels)}
label2id = {label:idx for idx, label in enumerate(labels)}

# Preprocess data
# As models like BERT don't expect text as direct input, but rather input_ids, etc., we tokenize the text using the tokenizer. 
# Here I'm using the AutoTokenizer API, which will automatically load the appropriate tokenizer based on the checkpoint on the hub.
# What's a bit tricky is that we also need to provide labels to the model. For multi-label text classification, this is a matrix of 
# shape (batch_size, num_labels). Also important: this should be a tensor of floats rather than integers, otherwise PyTorch' 
# BCEWithLogitsLoss (which the model will use) will complain, as explained here.

from transformers import AutoTokenizer
import numpy as np

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def preprocess_data(examples):
  # take a batch of texts
  text = examples["Tweet"]
  # encode them
  encoding = tokenizer(text, padding="max_length", truncation=True, max_length=128)
  # add labels
  labels_batch = {k: examples[k] for k in examples.keys() if k in labels}
  # create numpy array of shape (batch_size, num_labels)
  labels_matrix = np.zeros((len(text), len(labels)))
  # fill numpy array
  for idx, label in enumerate(labels):
    labels_matrix[:, idx] = labels_batch[label]

  encoding["labels"] = labels_matrix.tolist()
  
  return encoding

# Now we can apply this function to our dataset. This will take a while, as we're tokenizing all texts in the dataset.
encoded_dataset = dataset.map(preprocess_data, batched=True, remove_columns=dataset['train'].column_names)

# set the format of our data to PyTorch tensors.  This will turn the training, validation and test sets into standard PyTorch datasets.
encoded_dataset.set_format("torch")


# Define model
# Now we can define our model. Here I'm using the AutoModelForSequenceClassification API, which will automatically load the appropriate model 
# based on the checkpoint on the hub. I'm also setting the number of labels to the number of labels in our dataset, and setting the 
# output_attentions and output_hidden_states flags to True, so that we can access the attention weights and hidden states of the model later on.

from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", 
                                                           problem_type="multi_label_classification", 
                                                           num_labels=len(labels),
                                                           id2label=id2label,
                                                           label2id=label2id)


# Train the model
# Now we can train the model. I'm using the Trainer API, which is a high-level API for PyTorch that makes training models a lot easier.

batch_size = 8
metric_name = "f1"

from transformers import TrainingArguments, Trainer

args = TrainingArguments(
    f"bert-finetuned-sem_eval-english",
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model=metric_name,
    #push_to_hub=True,
)


# We are also going to compute metrics while training. For this, we need to define a compute_metrics function, that returns a dictionary 
# with the desired metric values.

from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from transformers import EvalPrediction
import torch
    
# source: https://jesusleal.io/2021/04/21/Longformer-multilabel-classification/
def multi_label_metrics(predictions, labels, threshold=0.5):
    # first, apply sigmoid on predictions which are of shape (batch_size, num_labels)
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    # next, use threshold to turn them into integer predictions
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1
    # finally, compute metrics
    y_true = labels
    f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
    roc_auc = roc_auc_score(y_true, y_pred, average = 'micro')
    accuracy = accuracy_score(y_true, y_pred)
    # return as dictionary
    metrics = {'f1': f1_micro_average,
               'roc_auc': roc_auc,
               'accuracy': accuracy}
    return metrics

def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, 
            tuple) else p.predictions
    result = multi_label_metrics(
        predictions=preds, 
        labels=p.label_ids)
    return result


#forward pass
outputs = model(input_ids=encoded_dataset['train']['input_ids'][0].unsqueeze(0), labels=encoded_dataset['train'][0]['labels'].unsqueeze(0))


# Now we can instantiate the Trainer class and start training. This will take a while, as we're training the model for 5 epochs.
trainer = Trainer(
    model,
    args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()
