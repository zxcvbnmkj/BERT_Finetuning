from os import path as osp

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from transformers import BertTokenizer
from main import calculate_metrics
from transformers import BertForSequenceClassification

dir_name = osp.dirname(__file__)

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

def eval_classification(y_true: pd.Series, y_pred: pd.Series, title=None):
    # acc P R F1 support 等指标的报告
    report = classification_report(y_true, y_pred, output_dict=True)
    metrics_df = pd.DataFrame(report).transpose()
    if title:
        print(f"\nMetrics by Class: ({title})")
    else:
        print("\nMetrics by Class:")
    print(metrics_df)

    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    classes = metrics_df.index.to_numpy()[:-3]
    cm_df = pd.DataFrame(cm, index=classes, columns=classes)
    print("Confusion Matrix:")
    print(cm_df)

if __name__ == '__main__':
    batch_size = 64
    df = pd.read_csv(f'{dir_name}/data/test_set.csv')
    sentences = df['text'].tolist()
    labels = df['label'].tolist()
    tokenizer = BertTokenizer.from_pretrained('chinese-bert-wwm')
    test_data = tokenizer(
        sentences,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors='pt'
    )
    prediction_dataset = TensorDataset(test_data['input_ids'], test_data['attention_mask'], torch.tensor(labels))
    prediction_dataloader = DataLoader(prediction_dataset, sampler=SequentialSampler(prediction_dataset),
                                       batch_size=batch_size)
    model = BertForSequenceClassification.from_pretrained(f'{dir_name}/bert_classifier')
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)
    predictions = []
    true_labels = []
    model.eval()
    for batch in prediction_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        with torch.no_grad():
            outputs = model(b_input_ids, attention_mask=b_input_mask)
        logits = outputs.logits.detach().cpu().numpy()
        label_ids = b_labels.cpu().numpy()

        predictions.extend(np.argmax(logits,axis=1))
        true_labels.extend(label_ids)
    # acc, p, r, f1 = calculate_metrics(logits, label_ids)
    # print(f"Text Metrics: {acc:.4f}, {p: 4f}, {r:4f},{f1:4f}")
    eval_classification(pd.Series(true_labels), pd.Series(predictions))

