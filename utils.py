import json
import os
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
import logging
import glob
from os import path as osp
from sklearn.metrics import precision_score, recall_score, f1_score
from torch import nn


def set_logger():
    logging.basicConfig(
        filename='finetuning_dp1_adamw.log',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        encoding='utf-8'
    )


# 把 json 形式数据转换为 df
def data_transform(dir_name):
    print("由 json 转换为 csv，数据形式转换中...")
    data_list = []
    json_files = glob.glob(osp.join(f"{dir_name}/json_files", "*.json"))
    for file_path in json_files:
        filename = os.path.basename(file_path)
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if filename.startswith('n'):
            label = 0
            data_part = [{'text': item['text'], 'label': label} for item in data]
        elif filename.startswith('p'):
            label = 1
            data_part = [{'text': item['text'].replace('\n', ' ').replace('\\', '').replace('\'', '').replace('\"', ''),
                          'label': label} for item in data]
        data_list.extend(data_part)
    df = pd.DataFrame(data_list)
    total_count = len(df)
    positive_count = len(df[df['label'] == 1])
    negative_count = len(df[df['label'] == 0])
    # 数据总数：272068
    # 正样本数 (label=1)：72068
    # 负样本数 (label=0)：200000
    print(f"数据总数：{total_count}")
    print(f"正样本数 (label=1)：{positive_count}")
    print(f"负样本数 (label=0)：{negative_count}")
    test_set_0 = df[df['label'] == 0].sample(n=250, random_state=42)  # 500 条测试集数据，正负样本各 250 条
    test_set_1 = df[df['label'] == 1].sample(n=250, random_state=42)
    test_set = pd.concat([test_set_0, test_set_1])
    train_set = df.drop(test_set.index)
    os.makedirs(f'{dir_name}/data')
    test_set.to_csv(f'{dir_name}/data/testset.csv', index=False, encoding='utf-8')
    train_set.to_csv(f'{dir_name}/data/trainset.csv', index=False, encoding='utf-8', escapechar='\\')
    return train_set


def calculate_metrics(preds, labels, threshold=0.5):
    preds = nn.functional.softmax(torch.tensor(preds), dim=-1).numpy()
    preds = (preds[:, 1] > threshold).astype(int).flatten()
    labels = labels.flatten()
    acc = np.sum(preds == labels) / len(labels)
    precision = precision_score(labels, preds, average='binary')
    recall = recall_score(labels, preds, average='binary')
    f1 = f1_score(labels, preds, average='binary')
    return acc, precision, recall, f1


def eval_classification(y_true: pd.Series, y_pred: pd.Series, title=None):
    # acc P R F1 support 等指标的报告
    report = classification_report(y_true, y_pred, output_dict=True)
    metrics_df = pd.DataFrame(report).transpose()
    if title:
        logging.info(f"\nMetrics by Class: ({title})")
    else:
        logging.info("\nMetrics by Class:")
    logging.info(metrics_df)
    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    classes = metrics_df.index.to_numpy()[:-3]
    cm_df = pd.DataFrame(cm, index=classes, columns=classes)
    logging.info("Confusion Matrix:")
    logging.info(cm_df)


def sentence_process(args, df):
    if args.task == 0:
        sentences = df['answer'].apply(lambda x: str(x)[-1200:]).tolist()
    elif args.task == 1:
        sentences = df.apply(concatenate_and_trim_token, axis=1).tolist()
    else:
        raise ValueError(f"不支持的任务类型: {args.task}，支持 0 或 1")
    return sentences


def concatenate_and_trim_token(row):
    answer = str(row['answer'])
    question = str(row['question'])
    combined_text = answer + '[SEP]' + question
    if len(combined_text) > 1200:
        return combined_text[-1200:]
    else:
        return combined_text