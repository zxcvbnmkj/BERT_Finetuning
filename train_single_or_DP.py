"""
`prohibit_parallel = True`: 在双卡 GPU 上只使用其中 1 张卡训练（默认使用第 1 张）
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
from utils import eval_classification, set_logger, data_transform

warnings.filterwarnings("ignore")

# 不能为 cuda:1 ，分布式训练会出现错误
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dir_name = osp.dirname(__file__)


def finetuning(epochs, max_patient, threshold, print_step):
    best_value = 0
    patient = 0
    for epoch_i in trange(epochs, desc="Epoch"):
        logging.info(f"当前是第{epoch_i}轮")
        logging.info("==========训练中=================")
        total_train_loss = 0
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
            total_train_loss += loss.item()
            # report 形式的指标
            train_true_labels.extend(label_ids)
            pred = nn.functional.softmax(torch.tensor(logits), dim=-1).numpy()
            pred = (pred[:, 1] > threshold).astype(int)
            train_predictions.extend(pred)
        avg_train_loss = total_train_loss / len(train_dataloader)
        eval_classification(pd.Series(train_true_labels), pd.Series(train_predictions), f"训练集_轮{epoch_i}")
        logging.info(f"\nEpoch {epoch_i + 1}/{epochs}")
        logging.info(f"Train loss: {avg_train_loss:.4f}")

        logging.info("========== 验证阶段 ==========")
        val_predictions = []
        val_true_labels = []
        model.eval()
        with torch.no_grad():
            for batch in val_dataloader:
                b_input_ids, b_input_mask, b_labels = tuple(t.to(device) for t in batch)
                outputs = model(b_input_ids,
                                attention_mask=b_input_mask)
                logits = outputs.logits.detach().cpu().numpy()
                label_ids = b_labels.to('cpu').numpy()
                # report 形式的指标
                val_true_labels.extend(label_ids)
                pred = nn.functional.softmax(torch.tensor(logits), dim=-1).numpy()
                pred = (pred[:, 1] > threshold).astype(int)
                val_predictions.extend(pred)
        report = eval_classification(pd.Series(val_true_labels), pd.Series(val_predictions), f"验证集_轮{epoch_i}")

        for name, param in model.named_parameters():
            if param is not None:
                param.data = param.data.contiguous()
        # 用于消融实验
        type = "dp1_adamw"
        # if report['macro avg']['f1-score'] > best_f1:
        #     best_f1 = report['macro avg']['f1-score']
        epoch_value=(report['0']['recall'] + report['1']['precision']) / 2.0
        if epoch_value > best_value:
            best_value = epoch_value
            patient = 0
            logging.info(f"存储最佳模型在：{dir_name}/{type}_best_{epoch_i}_bert_classifier")
            if hasattr(model, 'module'):
                model.module.save_pretrained(f'{dir_name}/{type}_best_{epoch_i}_bert_classifier')
            else:
                model.save_pretrained(f'{dir_name}/{type}_best_{epoch_i}_bert_classifier')
            tokenizer.save_pretrained(f'{dir_name}/{type}_best_{epoch_i}_bert_classifier')
        else:
            patient += 1
            logging.info(f"存储非最佳模型在：{dir_name}/{type}_{epoch_i}_bert_classifier")
            if hasattr(model, 'module'):
                model.module.save_pretrained(f'{dir_name}/{type}_{epoch_i}_bert_classifier')
            else:
                model.save_pretrained(f'{dir_name}/{type}_{epoch_i}_bert_classifier')
            tokenizer.save_pretrained(f'{dir_name}/{type}_{epoch_i}_bert_classifier')
        if patient == max_patient:
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=7)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--max_patient', type=int, default=3, help='最大容忍次数')
    parser.add_argument('--threshold', type=float, default=0.5, help='认定为正类（1）的阈值')
    parser.add_argument('--if_sub', action='store_true', help='是否使用子数据集训练与验证')
    parser.add_argument('--prohibit_parallel', action='store_true', help='即使有多张显卡也只用单卡训练')
    parser.add_argument('--sub_num', type=int, default=10, help='从原数据集中截取 sub_num 条数据')
    parser.add_argument('--mode', type=int, default=1, help='0: 给出的是`正样本.json`和`负样本.json`二者没有混合，此时需要把它们混合之后再分隔为训练集和测试集；'
                                                            '1: 给出的是训练集、测试集（验证集可选）')
    parser.add_argument('--task', type=int, default=1, help='0: 单句任务；'
                                                            '1: 双句任务，增加一个两个句子拼接的处理')
    parser.add_argument('--log_file', type=str, default='bert_finetuning.log', help='日志文件名字')
    args = parser.parse_args()
    # 只要在程序启动时调用了，则 logging.info() 全局有效，不必通过形参传递到函数中
    set_logger(args.log_file)

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
        logging.info("子数据集样本数：" + str(len(df)))
    # 使得序列 1 限制长度
    df['question'] = df['question'].apply(
        lambda x: str(x)[-200:]
    )
    if df_valid is not None:
        df_valid['question'] = df_valid['question'].apply(
            lambda x: str(x)[-200:]
        )
    # 日志只能有一个参数
    logging.info(f"训练集长度: {len(df)}")
    tokenizer = BertTokenizer.from_pretrained('/home/ubuntu/bert_classification/chinese-bert-wwm')
    tokenizer.truncation_side = "left"
    # sentences = sentence_process(args, df)
    if args.task == 1:
        encoded_inputs = tokenizer(
            df['question'],
            df['answer'],
            padding=True,
            truncation='only_second',
            max_length=512,
            return_tensors='pt')
    elif args.task == 0:
        encoded_inputs = tokenizer(
            df['answer'],
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
    # 用户没有给出验证集，则从训练集中划分出
    if df_valid is None:
        labels = df['label'].tolist()
        train_inputs, val_inputs, train_masks, val_masks, train_labels, val_labels = train_test_split(
            encoded_inputs['input_ids'],
            encoded_inputs['attention_mask'],
            labels,
            test_size=0.01,
            random_state=42
        )
    else:
        train_labels = df['label'].tolist()
        train_inputs = encoded_inputs['input_ids']
        train_masks = encoded_inputs['attention_mask']
        if args.if_sub:
            df_valid = df_valid.head(10)
        logging.info("验证集长度" + str(len(df_valid)))
        # sentences_valid = sentence_process(args, df_valid)
        val_labels = df_valid['label'].tolist()
        if args.task == 1:
            encoded_inputs_valid = tokenizer(
                df_valid['question'],
                df_valid['answer'],
                padding=True,
                truncation='only_second',
                max_length=512,
                return_tensors='pt'
            )
        elif args.task == 0:
            encoded_inputs_valid = tokenizer(
                df_valid['answer'],
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
        "/home/ubuntu/bert_classification/chinese-bert-wwm", num_labels=2).to(device)
    # 断点续训
    # f"{dir_name}/bert_classifier", num_labels=2).to(device)

    if not args.prohibit_parallel and torch.cuda.device_count() > 1:
        print(f"有 {torch.cuda.device_count()} 个GPU，使用分布式训练")
        model = nn.DataParallel(model)

    optimizer = AdamW(model.parameters(), lr=1e-5)

    finetuning(epochs=args.epochs, max_patient=args.max_patient, threshold=args.threshold, print_step=args.print_step)