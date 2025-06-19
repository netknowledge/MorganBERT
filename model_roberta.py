import numpy as np
import pandas as pd
import pickle
import time
from scipy import stats
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score, root_mean_squared_error, 
    roc_auc_score, precision_score, recall_score, f1_score, accuracy_score, balanced_accuracy_score
)

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, BertTokenizer, RobertaModel, RobertaForSequenceClassification, Trainer, TrainingArguments


class TextDataset(Dataset):
    def __init__(self, text_list, label_list, tokenizer, task, max_length=512):
        assert task in ['regression', 'classification']
        self.text_list = text_list
        self.label_list = label_list
        self.tokenizer = tokenizer
        self.task = task
        self.max_length = max_length

    def __len__(self):
        return len(self.text_list)

    def __getitem__(self, idx):
        text = self.text_list[idx]
        label = self.label_list[idx]
        # Tokenizing the text
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        # Extract the single encoding from the batch
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': torch.tensor(label, dtype=torch.float) if self.task == 'regression' else torch.tensor(label, dtype=torch.long)
        }


def compute_metrics_reg(eval_pred):
    """"""
    predictions, labels = eval_pred # (N, 1); (N,)
    predictions = predictions[:,0]
    return {
        'mse': mean_squared_error(labels, predictions), 
        'mae': mean_absolute_error(labels, predictions), 
        'r2':  r2_score(labels, predictions), 
        'rmse':root_mean_squared_error(labels, predictions), 
        'pcc': stats.pearsonr(predictions, labels)[0]
    }


def softmax(x):
    """"""
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)


def compute_metrics_cls(eval_pred):
    """"""
    predictions, labels = eval_pred # (N,2); (N,)
    y_score = softmax(predictions)[:,1] # normalize into prob
    y_pred = np.argmax(predictions, axis=-1)
    return {
        'auc': roc_auc_score(labels, y_score), 
        'prec': precision_score(labels, y_pred), 
        'recall': recall_score(labels, y_pred), 
        'f1': f1_score(labels, y_pred), 
        'accuracy': accuracy_score(labels, y_pred), 
        'balanced_accuracy': balanced_accuracy_score(labels, y_pred), 
    }


def run_roberta_like_train_test(model_path, tokenizer, train_text_list, train_label_list, test_text_list, test_label_list, task, output_dir, epochs):
    """"""
    if task == 'regression':
        nlabel = 1
        metrics_func = compute_metrics_reg
    elif task == 'classification':
        nlabel = 2
        metrics_func = compute_metrics_cls
    model = RobertaForSequenceClassification.from_pretrained(model_path, num_labels=nlabel)
    train_dataset = TextDataset(train_text_list, train_label_list, tokenizer, task)
    training_args = TrainingArguments(
        output_dir=output_dir, 
        num_train_epochs=epochs, 
        per_device_train_batch_size=8, 
        per_device_eval_batch_size=16, 
        weight_decay=0.01, 
        logging_steps=500, 
        save_total_limit=2
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        compute_metrics=metrics_func
    )
    trainer.train()
    test_dataset = TextDataset(test_text_list, test_label_list, tokenizer, task)
    return trainer.evaluate(test_dataset)


def run_roberta_like(model_group, model_name, text_series, y_series, train_test_index, task_name, task, epochs=30):
    """"""
    if model_group == 'chemberta':
        model_path = '/mnt/data2/morgan-bert/baseline_models/' + model_name
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    elif model_group == 'morganbert':
        model_path = '/mnt/data2/morgan-bert/MorganBERT_models/' + model_name
        tokenizer = BertTokenizer.from_pretrained(model_path + '/vocab')
    output_dir = '/mnt/data2/morgan-bert_v2/tmp_models_%s/%s_%s' % (model_group, model_name, task_name)
    results = []
    for train_index, test_index in train_test_index:
        start = time.time()
        eval_result = run_roberta_like_train_test(model_path, tokenizer, 
                                                  text_series.loc[train_index].tolist(), y_series.loc[train_index].tolist(), 
                                                  text_series.loc[test_index].tolist(), y_series.loc[test_index].tolist(), 
                                                  task, output_dir, epochs)
        end = time.time()
        eval_result['total_runtime_sec'] = (end - start)
        results.append(eval_result)
    return pd.DataFrame(results)
