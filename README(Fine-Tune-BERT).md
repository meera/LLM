# Toxic Comment Classification with BERT

This repository contains a Python script that uses the BERT model to classify comments from the Jigsaw Toxic Comment Classification Challenge as toxic or non-toxic.

## Installation

Use pip to install the required packages:

```bash
pip install transformers pandas scikit-learn torch numpy
```

## Usage

Firstly, download the training dataset (train.csv) from the [Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data) and place it in the appropriate path. Modify the path in the script if needed.

Next, you can run the Python script. The script will train the model, evaluate it, and save the trained model.

## How it Works

1. The script reads the data from the CSV file and selects the first 1000 rows of the 'comment_text' and 'toxic' columns.

2. It uses the 'bert-base-uncased' tokenizer and model from the transformers library.

3. The data is split into training and validation sets, and the text is tokenized.

4. A custom Dataset class is defined and used to create torch datasets.

5. A function, `compute_metrics`, is defined to compute the accuracy, precision, recall, and F1 score.

6. The TrainingArguments are set and the Trainer class from transformers is used to train and evaluate the model.

7. Finally, the trained model is saved.

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

[MIT](https://choosealicense.com/licenses/mit/)
