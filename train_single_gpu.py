"""
在双卡 GPU 上只使用其中 1 张卡训练
调整 0/1 类别上的阈值，其实就是在调整模型预测样本时的偏好，相当于调整的了焦点损失中的权重值
"""
# -*- coding: utf-8 -*-
import argparse
import glob
import logging
import warnings
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
from transformers import BertForSequenceClassification
from torch.optim import AdamW
from tqdm import trange
import pandas as pd
from os import path as osp
from utils import eval_classification, set_logger, calculate_metrics, data_transform, sentence_process

warnings.filterwarnings("ignore")

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
dir_name = osp.dirname(__file__)


def finetuning(epochs, max_patient, threshold, print_step):
    best_f1 = 0
    patient = 0
    for epoch_i in trange(epochs, desc="Epoch"):
        logging.info(f"当前是第{epoch_i}轮")
        logging.info("==========训练中=================")
        total_train_loss, total_train_acc, total_train_p, total_train_r, total_train_f1 = 0, 0, 0, 0, 0
        train_predictions = []
        train_true_labels = []
        for index, batch in enumerate(train_dataloader):
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
            acc, p, r, f1 = calculate_metrics(logits, label_ids,threshold)
            total_train_acc += acc
            total_train_p += p
            total_train_r += r
            total_train_f1 += f1
            total_train_loss += loss.item()
            # report 形式的指标
            train_true_labels.extend(label_ids)
            pred = nn.functional.softmax(torch.tensor(logits), dim=-1).numpy()
            pred = (pred[:, 1] > threshold).astype(int)
            train_predictions.extend(pred)
            if index % print_step == 0:
                logging.info(f"训练集{epoch_i} - 批次 {index} 的指标：acc: {acc},p: {p},r: {r},f1: {f1}")
        avg_train_loss = total_train_loss / len(train_dataloader)
        avg_train_acc = total_train_acc / len(val_dataloader)
        avg_train_p = total_train_p / len(val_dataloader)
        avg_train_r = total_train_r / len(val_dataloader)
        avg_train_f1 = total_train_f1 / len(val_dataloader)
        eval_classification(pd.Series(train_true_labels), pd.Series(train_predictions), f"训练集_轮{epoch_i}")
        logging.info(f"\nEpoch {epoch_i + 1}/{epochs}")
        logging.info(f"Train loss: {avg_train_loss:.4f}")
        logging.info(f"Train Metrics: {avg_train_acc:.4f}, {avg_train_p: 4f}, {avg_train_r:4f},{avg_train_f1:4f}")

        logging.info("========== 验证阶段 ==========")
        val_predictions = []
        val_true_labels = []
        model.eval()
        total_eval_acc, total_eval_p, total_eval_r, total_eval_f1 = 0, 0, 0, 0
        with torch.no_grad():
            for batch in val_dataloader:
                b_input_ids, b_input_mask, b_labels = tuple(t.to(device) for t in batch)
                outputs = model(b_input_ids,
                                attention_mask=b_input_mask)
                logits = outputs.logits.detach().cpu().numpy()
                label_ids = b_labels.to('cpu').numpy()
                acc, p, r, f1 = calculate_metrics(logits, label_ids,threshold)
                total_eval_acc += acc
                total_eval_p += p
                total_eval_r += r
                total_eval_f1 += f1
                # report 形式的指标
                val_true_labels.extend(label_ids)
                pred = nn.functional.softmax(torch.tensor(logits), dim=-1).numpy()
                pred = (pred[:, 1] > threshold).astype(int)
                val_predictions.extend(pred)
                if index % print_step == 0:
                    logging.info(f"验证集{epoch_i} - 批次 {index} 的指标：acc: {acc},p: {p},r: {r},f1: {f1}")
        avg_val_accuracy = total_eval_acc / len(val_dataloader)
        avg_val_p = total_eval_p / len(val_dataloader)
        avg_val_r = total_eval_r / len(val_dataloader)
        avg_val_f1 = total_eval_f1 / len(val_dataloader)
        eval_classification(pd.Series(train_true_labels), pd.Series(train_predictions), f"验证集_轮{epoch_i}")
        logging.info(f"Validation Metrics: {avg_val_accuracy:.4f}, {avg_val_p: 4f}, {avg_val_r:4f},{avg_val_f1:4f}")

        for name, param in model.named_parameters():
            if param is not None:
                param.data = param.data.contiguous()
        if avg_val_f1 > best_f1:
            best_f1 = avg_val_f1
            patient = 0
            model.save_pretrained(f'{dir_name}/best_{epoch_i}_bert_classifier')
            tokenizer.save_pretrained(f'{dir_name}/best_{epoch_i}_bert_classifier')
        else:
            patient += 1
            model.save_pretrained(f'{dir_name}/{epoch_i}_bert_classifier')
            tokenizer.save_pretrained(f'{dir_name}/{epoch_i}_bert_classifier')
        if patient == max_patient:
            break

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=7)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--max_patient', type=int, default=3, help='最大容忍次数')
    parser.add_argument('--threshold', type=float, default=0.5, help='认定为正类（1）的阈值')
    parser.add_argument('--print_step', type=int, default=150, help='多少批次打印一次批结果')
    parser.add_argument('--if_sub', action='store_true', help='是否使用子数据集训练与验证')
    parser.add_argument('--sub_num', type=int, default=10, help='从原数据集中截取 sub_num 条数据')
    parser.add_argument('--mode', type=int, default=1, help='0: 给出的是`正样本.json`和`负样本.json`二者没有混合，此时需要把它们混合之后再分隔为训练集和测试集；'
                                                            '1: 给出的是训练集、测试集（验证集可选）')
    parser.add_argument('--task', type=int, default=1, help='0: 单句任务；'
                                                            '1: 双句任务，增加一个两个句子拼接的处理')
    args = parser.parse_args()
    # 只要在程序启动时调用了，则 logging.info() 全局有效，不必通过形参传递到函数中
    set_logger()

    if args.mode == 0 and not osp.exists(f"{dir_name}/data_acc_new/trainset.csv"):
        df = data_transform()

    # 获取 data 文件夹下第一个文件的后缀
    data_files = glob.glob(osp.join(f"{dir_name}/data_acc_new", "*"))
    if not data_files:
        raise FileNotFoundError(f"data 文件夹下没有文件")
    file_ext = osp.splitext(data_files[0])[1].lower()
    train_file = osp.join(f"{dir_name}/data_acc_new", f"trainset{file_ext}")
    valid_file = osp.join(f"{dir_name}/data_acc_new", f"validset{file_ext}")
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
        logging.info("子数据集样本数：" + len(df))

    logging.info("训练集长度", len(df))
    tokenizer = BertTokenizer.from_pretrained('/data/bert_check_completed/chinese-bert-wwm')
    tokenizer.truncation_side = "left"
    sentences = sentence_process(args, df)

    # 用户没有给出验证集，则从训练集中划分出
    if df_valid is None:
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
        if args.if_sub:
            df_valid=df_valid.head(10)
        logging.info("验证集长度" + len(df_valid))
        sentences_valid = sentence_process(args, df_valid)
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
    train_dataloader = DataLoader(train_sample, batch_size=args.batch_size, shuffle=True)
    val_sample = TensorDataset(val_data['input_ids'], val_data['attention_mask'], val_data['labels'])
    val_dataloader = DataLoader(val_sample, batch_size=args.batch_size, shuffle=True)
    model = BertForSequenceClassification.from_pretrained(
        # "/data/bert_check_completed/chinese-bert-wwm", num_labels=2).to(device)
        # 断点续训
        f"{dir_name}/bert_classifier", num_labels=2).to(device)

    optimizer = AdamW(model.parameters(), lr=1e-5)

    finetuning(epochs=args.epochs, max_patient=args.max_patient, threshold=args.threshold, print_step=args.print_step)