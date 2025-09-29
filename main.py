import argparse
import glob
import json
import logging
import os
import torch
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score, f1_score
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
from transformers import BertForSequenceClassification
from torch.optim import AdamW
from tqdm import trange
import numpy as np
import pandas as pd
from os import path as osp

if torch.backends.mps.is_available():
    # MAC系统
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

dir_name = osp.dirname(__file__)


def set_logger():
    logging.basicConfig(
        filename='finetuning.log',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        encoding='utf-8'
    )

# 把 json 形式数据转换为 df
def data_transform():
    print("数据形式转换中...")
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
    test_set.to_csv(f'{dir_name}/data/test_set.csv', index=False, encoding='utf-8')
    train_set.to_csv(f'{dir_name}/data/train_set.csv', index=False, encoding='utf-8', escapechar='\\')
    return train_set


def calculate_metrics(preds, labels):
    preds = np.argmax(preds, axis=1).flatten()
    labels = labels.flatten()
    acc = np.sum(preds == labels) / len(labels)
    precision = precision_score(labels, preds, average='binary')  # 二分类
    recall = recall_score(labels, preds, average='binary')
    f1 = f1_score(labels, preds, average='binary')
    return acc, precision, recall, f1


def finetuning(epochs, max_patient):
    set_logger()
    best_f1 = 0
    patient = 0
    for epoch_i in trange(epochs, desc="Epoch"):
        # ========== 训练阶段 ==========
        print(f"当前是第{epoch_i}轮")
        print("训练中")
        total_train_loss, total_train_acc, total_train_p, total_train_r, total_train_f1 = 0, 0, 0, 0, 0
        index = 0
        for batch in train_dataloader:
            print("当前批次是", index)
            index += 1
            b_input_ids, b_input_mask, b_labels = tuple(t.to(device) for t in batch)
            model.zero_grad()
            outputs = model(b_input_ids,
                            attention_mask=b_input_mask,
                            labels=b_labels)
            loss = outputs.loss.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            logits = outputs.logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            acc, p, r, f1 = calculate_metrics(logits, label_ids)
            total_train_acc += acc
            total_train_p += p
            total_train_r += r
            total_train_f1 += f1
            total_train_loss += loss.item()
            print(f"本批次{acc},{p},{r},{f1}")
            logging.info(f"本批次{acc},{p},{r},{f1}")
        avg_train_loss = total_train_loss / len(train_dataloader)
        avg_train_acc = total_train_acc / len(val_dataloader)
        avg_train_p = total_train_p / len(val_dataloader)
        avg_train_r = total_train_r / len(val_dataloader)
        avg_train_f1 = total_train_f1 / len(val_dataloader)
        logging.info(f"\nEpoch {epoch_i + 1}/{epochs}")
        logging.info(f"Train loss: {avg_train_loss:.4f}")
        logging.info(f"Train Metrics: {avg_train_acc:.4f}, {avg_train_p: 4f}, {avg_train_r:4f},{avg_train_f1:4f}")

        # ========== 验证阶段 ==========
        print("验证中")
        model.eval()
        total_eval_acc, total_eval_p, total_eval_r, total_eval_f1 = 0, 0, 0, 0
        with torch.no_grad():
            for batch in val_dataloader:
                b_input_ids, b_input_mask, b_labels = tuple(t.to(device) for t in batch)
            outputs = model(b_input_ids,
                            attention_mask=b_input_mask)
            logits = outputs.logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            acc, p, r, f1 = calculate_metrics(logits, label_ids)
            total_eval_acc += acc
            total_eval_p += p
            total_eval_r += r
            total_eval_f1 += f1
            print(f"本批次{acc},{p},{r},{f1}")
            logging.info(f"本批次{acc},{p},{r},{f1}")
        avg_val_accuracy = total_eval_acc / len(val_dataloader)
        avg_val_p = total_eval_p / len(val_dataloader)
        avg_val_r = total_eval_r / len(val_dataloader)
        avg_val_f1 = total_eval_f1 / len(val_dataloader)
        logging.info(f"Validation Metrics: {avg_val_accuracy:.4f}, {avg_val_p: 4f}, {avg_val_r:4f},{avg_val_f1:4f}")

        if avg_val_f1 > best_f1:
            best_f1 = avg_val_f1
            patient = 0
            for name, param in model.named_parameters():
                if param is not None:
                    param.data = param.data.contiguous()
            model.module.save_pretrained(f'{dir_name}/clean_bert_classifier')
        else:
            patient += 1
        if patient == max_patient:
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=7)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--max_patient', type=int, default=2, help='最大容忍次数')
    parser.add_argument('--if_sub', action='store_true', help='是否使用子数据集训练与验证')
    parser.add_argument('--sub_num', type=int, default=10, help='从原数据集中截取 sub_num 条数据')
    args = parser.parse_args()

    if not osp.exists(f"{dir_name}/data"):
        df = data_transform()
    else:
        df = pd.read_csv(f"{dir_name}/data_clean/train_set.csv")

    # 仅取 sub_num 条测试代码是否正确
    # df = df.head(args.sub_num)
    if args.if_sub:
        df_0 = df[df['label'] == 0].head(int(args.sub_num / 2.0))
        df_1 = df[df['label'] == 1].head(int(args.sub_num / 2.0))
        df = pd.concat([df_0, df_1]).sample(frac=1).reset_index(drop=True)
        print("子数据集样本数：", len(df))
    sentences = df['text'].tolist()
    labels = df['label'].tolist()
    tokenizer = BertTokenizer.from_pretrained('chinese-bert-wwm')
    encoded_inputs = tokenizer(
        sentences,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors='pt'
    )
    train_inputs, val_inputs, train_masks, val_masks, train_labels, val_labels = train_val_data = train_test_split(
        encoded_inputs['input_ids'],
        encoded_inputs['attention_mask'],
        labels,
        test_size=0.01,
        random_state=42
    )
    train_data = {
        'input_ids': train_inputs.clone().detach(),
        'attention_mask': train_masks.clone().detach(),
        'labels': torch.tensor(train_labels)
    }
    val_data = {
        'input_ids': val_inputs.clone().detach(),
        'attention_mask': val_masks.clone().detach(),
        'labels': torch.tensor(val_labels)
    }
    train_sample = TensorDataset(train_data['input_ids'], train_data['attention_mask'], train_data['labels'])
    train_sampler = RandomSampler(train_sample)
    train_dataloader = DataLoader(train_sample, sampler=train_sampler, batch_size=args.batch_size)
    val_sample = TensorDataset(val_data['input_ids'], val_data['attention_mask'], val_data['labels'])
    val_sampler = RandomSampler(val_sample)
    val_dataloader = DataLoader(val_sample, sampler=val_sampler, batch_size=args.batch_size)
    model = BertForSequenceClassification.from_pretrained("chinese-bert-wwm", num_labels=2).to(device)
    if torch.cuda.device_count() > 1:
        print(f"有 {torch.cuda.device_count()} 个GPU")
        model = nn.DataParallel(model)

    # 为BERT等Transformer模型设置分组参数优化，主要目的是对不同类型的参数应用不同的权重衰减（weight decay）
    # 获取所有参数
    param_optimizer = list(model.named_parameters())
    # 不需要权重衰减的部分
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parametes = [
        # 第一组：需要权重衰减的参数
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.1},
        # 第二组：不需要衰减的参数。不能省略，因为需要把参数添加到优化器中
        {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parametes, lr=1e-5)

    finetuning(epochs=args.epochs, max_patient=args.max_patient)
