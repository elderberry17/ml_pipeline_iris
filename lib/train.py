import json
import os
import pickle
import random
from functools import partial

import numpy as np
import yaml
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
import mlflow

mlflow.set_tracking_uri('http://158.160.11.51:90/')
mlflow.set_experiment('elderberry17')

RANDOM_SEED = 1

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

METRICS = {
    'recall': partial(recall_score, average='macro'),
    'precision': partial(precision_score, average='macro'),
    'accuracy': accuracy_score,
}


def save_dict(data: dict, filename: str):
    with open(filename, 'w') as f:
        json.dump(data, f)


def load_dict(filename: str):
    with open(filename, 'r') as f:
        return json.load(f)


def train_model(model_type, x, y):
    model = model_type()
    model.fit(x, y)
    return model


def train():
    with open('params.yaml', 'rb') as f:
        params_data = yaml.safe_load(f)

    NAME_TO_MODEL = {
        'LogisticRegression': LogisticRegression,
        'DecisionTreeClassifier': DecisionTreeClassifier,
        'RandomForestClassifier': RandomForestClassifier, 
        'KNeighborsClassifier': KNeighborsClassifier,
    }

    config = params_data['train']
    task_dir = 'data/train'

    model_type = NAME_TO_MODEL[config['model']]
    data = load_dict('data/features_preparation/data.json')

    model = train_model(model_type, data['train_x'], data['train_y'])

    preds = model.predict(data['train_x'])

    metrics = {}
    for metric_name in params_data['eval']['metrics']:
        metrics[metric_name] = METRICS[metric_name](data['train_y'], preds)

    clf_report = classification_report(data['train_y'], preds)

    if not os.path.exists(task_dir):
        os.mkdir(task_dir)

    save_dict(clf_report, os.path.join(task_dir, 'clf_report.json'))
    save_dict(metrics, os.path.join(task_dir, 'metrics.json'))


    with open(f'data/train/model.pkl', 'wb') as f:
        pickle.dump(model, f)

    params = {}
    for i in params_data.values():
        params.update(i)

    params['run_type'] = 'train'

    print(f'train params - {params}')
    print(f'train metrics - {metrics}')

    mlflow.log_params(params)
    mlflow.log_metrics(metrics)
    mlflow.log_artifact(os.path.join(task_dir, 'clf_report.json'))

    # с catboost'ом надо иначе
    mlflow.sklearn.log_model(model, "model.pkl")


if __name__ == '__main__':
    train()
