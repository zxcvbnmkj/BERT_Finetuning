# -*- coding: utf-8 -*-
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
    # 用于 MAC系统
    # mps: Metal Performance Shaders，Apple Silicon（M1/M2/M3 等）的 GPU 加速框架
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


def sentence_process(args, df):
    if args.task == 0:
        sentences = df['text'].apply(lambda x: str(x)[-512:]).tolist()
    elif args.task == 1:
        sentences = df.apply(concatenate_and_trim, axis=1).tolist()
    else:
        raise ValueError(f"不支持的任务类型: {args.task}，支持 0 或 1")
    return sentences


# 把 json 形式数据转换为 df
def data_transform():
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
            print(f"本批次指标：acc: {acc},p: {p},r: {r},f1: {f1}")
            logging.info(f"本批次指标：acc: {acc},p: {p},r: {r},f1: {f1}")
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
                print(f"本批次指标：acc: {acc},p: {p},r: {r},f1: {f1}")
                logging.info(f"本批次指标：acc: {acc},p: {p},r: {r},f1: {f1}")
        avg_val_accuracy = total_eval_acc / len(val_dataloader)
        avg_val_p = total_eval_p / len(val_dataloader)
        avg_val_r = total_eval_r / len(val_dataloader)
        avg_val_f1 = total_eval_f1 / len(val_dataloader)
        logging.info(f"Validation Metrics: {avg_val_accuracy:.4f}, {avg_val_p: 4f}, {avg_val_r:4f},{avg_val_f1:4f}")

        if avg_val_f1 > best_f1:
            best_f1 = avg_val_f1
            patient = 0
            # 强制确保模型参数的内存布局是连续，用于防止错误 "你在保持一个非连续的张量"
            # ValueError: You are trying to save a non contiguous tensor: `bert.encoder.layer.0.attention.self.query.weight` which is not allowed. It either means you are trying to save tensors which are reference of each other in which case it's recommended to save only the full tensors, and reslice at load time, or simply call `.contiguous()` on your tensor to pack it before saving.
            for name, param in model.named_parameters():
                if param is not None:
                    param.data = param.data.contiguous()
            # 如果使用了分布式训练
            if hasattr(model, 'module'):
                model.module.save_pretrained(f'{dir_name}/bert_classifier')
            else:
                model.save_pretrained(f'{dir_name}/bert_classifier')
            # 保存分词器，并放到模型文件夹内。这样在推理的时候就完全不需要用到预训练模型了，只需要一个微调后模型即可
            tokenizer.save_pretrained(f'{dir_name}/bert_classifier')
        else:
            patient += 1
        if patient == max_patient:
            break


def concatenate_and_trim(row):
    combined_text = row['answer'] + '[SEP]' + row['question']
    return combined_text[-512:]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=7)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--max_patient', type=int, default=2, help='最大容忍次数')
    parser.add_argument('--if_sub', action='store_true', help='是否使用子数据集训练与验证')
    parser.add_argument('--sub_num', type=int, default=10, help='从原数据集中截取 sub_num 条数据')
    parser.add_argument('--mode', type=int, default=1, help='0: 给出的是`正样本.json`和`负样本.json`二者没有混合，此时需要把它们混合之后再分隔为训练集和测试集；'
                                                            '1: 给出的是训练集、测试集（验证集可选）')
    parser.add_argument('--task', type=int, default=1, help='0: 单句任务；'
                                                            '1: 双句任务，增加一个两个句子拼接的处理')
    args = parser.parse_args()

    if args.mode == 0 and not osp.exists(f"{dir_name}/data/trainset.csv"):
        df = data_transform()

    # 获取 data 文件夹下第一个文件的后缀
    data_files = glob.glob(osp.join(f"{dir_name}/data", "*"))
    if not data_files:
        raise FileNotFoundError(f"data 文件夹下没有文件")
    file_ext = osp.splitext(data_files[0])[1].lower()
    train_file = osp.join(f"{dir_name}/data", f"trainset{file_ext}")
    valid_file = osp.join(f"{dir_name}/data", f"validset{file_ext}")
    df_valid = None
    if file_ext == '.json':
        df = pd.read_json(train_file)
        if osp.exists(valid_file):
            df_valid = pd.read_json(valid_file)
    elif file_ext == '.csv':
        df = pd.read_csv(train_file)
        if osp.exists(valid_file):
            df_valid = pd.read_csv(valid_file)
    else:
        raise ValueError(f"不支持的文件格式: {file_ext}，仅支持 .json 或 .csv")

    # 仅取 sub_num 条测试代码是否正确
    # df = df.head(args.sub_num)
    if args.if_sub:
        df_0 = df[df['label'] == 0].head(int(args.sub_num / 2.0))
        df_1 = df[df['label'] == 1].head(int(args.sub_num / 2.0))
        df = pd.concat([df_0, df_1]).sample(frac=1).reset_index(drop=True)
        print("子数据集样本数：", len(df))

    tokenizer = BertTokenizer.from_pretrained('/Users/nowcoder/workspace/bert_classification/chinese-bert-wwm')

    # 用户没有给出验证集，则从测试集中划分出
    if df_valid is None:
        sentences = sentence_process(args, df)
        labels = df['label'].tolist()
        encoded_inputs = tokenizer(
            sentences,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        train_inputs, val_inputs, train_masks, val_masks, train_labels, val_labels = train_test_split(
            encoded_inputs['input_ids'],
            encoded_inputs['attention_mask'],
            labels,
            test_size=0.01,
            random_state=42
        )
    else:
        sentences = sentence_process(args, df)
        train_labels = df['label'].tolist()
        encoded_inputs = tokenizer(
            sentences,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        train_inputs = encoded_inputs['input_ids']
        train_masks = encoded_inputs['attention_mask']

        sentences_valid = sentence_process(args, df)
        val_labels = df_valid['label'].tolist()
        encoded_inputs_valid = tokenizer(
            sentences_valid,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        val_inputs = encoded_inputs_valid['input_ids']
        val_masks = encoded_inputs_valid['attention_mask']

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
    # 这个类继承自 Dataset ,它只是以元组的形式返回输入的各个参数而已。当数据集逻辑并不复杂的时候，可以直接使用它，从而避免自定义 Dataset
    train_sample = TensorDataset(train_data['input_ids'], train_data['attention_mask'], train_data['labels'])
    # 使用了 RandomSampler 包裹数据类，使得每一轮都会打乱其中的样本
    # 但是这种写法并没有直接在 DataLoader 里面使用 shuffle 那么简单
    # train_sampler = RandomSampler(train_sample)
    # train_dataloader = DataLoader(train_sample, sampler=train_sampler, batch_size=args.batch_size)
    train_dataloader = DataLoader(train_sample, sampler=train_sample, batch_size=args.batch_size, shuffle=True)
    val_sample = TensorDataset(val_data['input_ids'], val_data['attention_mask'], val_data['labels'])
    # val_sampler = RandomSampler(val_sample)
    val_dataloader = DataLoader(val_sample, sampler=val_sample, batch_size=args.batch_size, shuffle=True)
    model = BertForSequenceClassification.from_pretrained(
        "/Users/nowcoder/workspace/bert_classification/chinese-bert-wwm", num_labels=2).to(device)
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
