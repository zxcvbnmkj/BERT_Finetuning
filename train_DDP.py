# -*- coding: utf-8 -*-
import argparse
import glob
import json
import logging
import os
import warnings
import torch
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score, f1_score
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import TensorDataset, DataLoader, DistributedSampler
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
from transformers import BertForSequenceClassification
from torch.optim import AdamW
from tqdm import trange
import numpy as np
import pandas as pd
from os import path as osp
from torch import distributed as dist
warnings.filterwarnings("ignore")

local_rank = int(os.environ.get("LOCAL_RANK", 0))
# 进程数，等于使用到了 GPU 数目
world_size = int(os.environ.get("WORLD_SIZE", 1))
# 当前是否主进程
is_main_process = (local_rank <= 0)

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
        filename='finetuning_DDP.log',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        encoding='utf-8'
    )


def sentence_process(args, df):
    if args.task == 0:
        sentences = df['answer'].apply(lambda x: str(x)[-512:]).tolist()
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


def calculate_metrics(preds, labels, threshold=0.8):
    # 默认阈值为 0
    # preds = np.argmax(preds, axis=1).flatten()
    # 可以设置阈值，当 1 标签的值大于 0.8 时才认为结束了
    preds = nn.functional.softmax(torch.tensor(preds), dim=-1).numpy()
    preds = (preds[:, 1] > threshold).astype(int).flatten()
    labels = labels.flatten()
    acc = np.sum(preds == labels) / len(labels)
    precision = precision_score(labels, preds, average='binary')
    recall = recall_score(labels, preds, average='binary')
    f1 = f1_score(labels, preds, average='binary')
    return acc, precision, recall, f1


def finetuning(epochs, max_patient):
    best_f1 = 0
    patient = 0
    for epoch_i in trange(epochs, desc="Epoch"):
        # ========== 训练阶段 ==========
        if world_size > 1:
            train_sampler.set_epoch(epoch_i)
        if is_main_process:
            logging.info(f"当前是第{epoch_i}轮")
            logging.info("================训练中==================")
        total_train_loss, total_train_acc, total_train_p, total_train_r, total_train_f1 = 0, 0, 0, 0, 0
        for index, batch in enumerate(train_dataloader):
            b_input_ids, b_input_mask, b_labels = tuple(t.to(device) for t in batch)
            model.zero_grad()
            outputs = model(b_input_ids,
                            attention_mask=b_input_mask,
                            labels=b_labels)
            loss = outputs.loss
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
            if is_main_process and index % 100 == 0:
                logging.info(f"训练集{epoch_i} - 批次 {index} 的指标：acc: {acc},p: {p},r: {r},f1: {f1}")
        avg_train_loss = total_train_loss / len(train_dataloader)
        avg_train_acc = total_train_acc / len(train_dataloader)
        avg_train_p = total_train_p / len(train_dataloader)
        avg_train_r = total_train_r / len(train_dataloader)
        avg_train_f1 = total_train_f1 / len(train_dataloader)
        # 同步所有进程的指标 ，若不加则只能得到 主进程上面的批次数量数据
        if world_size > 1:
            # 将指标转换为tensor
            train_metrics = torch.tensor([avg_train_loss, avg_train_acc, avg_train_p, avg_train_r, avg_train_f1],
                                         device=device)
            # 同步
            dist.all_reduce(train_metrics)
            # 均值
            train_metrics /= world_size
            avg_train_loss, avg_train_acc, avg_train_p, avg_train_r, avg_train_f1 = train_metrics.cpu().numpy()
        if is_main_process:
            logging.info(f"\nEpoch {epoch_i + 1}/{epochs}")
            logging.info(f"Train loss: {avg_train_loss:.4f}")
            logging.info(f"Train Metrics: {avg_train_acc:.4f}, {avg_train_p: 4f}, {avg_train_r:4f},{avg_train_f1:4f}")

        # ========== 验证阶段 ==========
        if is_main_process:
            logging.info("==============验证中===============")
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
                if is_main_process and index % 100 == 0:
                    logging.info(f"验证集 - 批次 {index} 指标：acc: {acc},p: {p},r: {r},f1: {f1}")
        avg_val_accuracy = total_eval_acc / len(val_dataloader)
        avg_val_p = total_eval_p / len(val_dataloader)
        avg_val_r = total_eval_r / len(val_dataloader)
        avg_val_f1 = total_eval_f1 / len(val_dataloader)
        if world_size > 1:
            val_metrics = torch.tensor([avg_val_accuracy, avg_val_p, avg_val_r, avg_val_f1], device=device)
            dist.all_reduce(val_metrics)
            val_metrics /= world_size
            avg_val_accuracy, avg_val_p, avg_val_r, avg_val_f1 = val_metrics.cpu().numpy()
        if is_main_process:
            logging.info(f"Validation Metrics: {avg_val_accuracy:.4f}, {avg_val_p: 4f}, {avg_val_r:4f},{avg_val_f1:4f}\n本轮实验结束\n\n\n")

        if avg_val_f1 > best_f1:
            if is_main_process:
                logging.info(f"存储当前最佳模型，属于轮{epoch_i}，它验证集 f1 为{avg_val_f1}")
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
        else:
            patient += 1
        if patient == max_patient:
            break
    # 保存分词器，并放到模型文件夹内。这样在推理的时候就完全不需要用到预训练模型了，只需要一个微调后模型即可
    if is_main_process:
        logging.info(f"存储分词器")
        tokenizer.save_pretrained(f'{dir_name}/bert_classifier')

    # 清理分布式资源
    if world_size > 1:
        dist.destroy_process_group()


def concatenate_and_trim(row):
    combined_text = row['answer'] + '[SEP]' + row['question']
    return combined_text


if __name__ == '__main__':
    set_logger()
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=80)
    parser.add_argument('--max_patient', type=int, default=1, help='最大容忍次数')
    parser.add_argument('--if_sub', action='store_true', help='是否使用子数据集训练与验证')
    parser.add_argument('--sub_num', type=int, default=10, help='从原数据集中截取 sub_num 条数据')
    parser.add_argument('--mode', type=int, default=1, help='0: 给出的是`正样本.json`和`负样本.json`二者没有混合，此时需要把它们混合之后再分隔为训练集和测试集；'
                                                            '1: 给出的是训练集、测试集（验证集可选）')
    parser.add_argument('--task', type=int, default=1, help='0: 单句任务；'
                                                            '1: 双句任务，增加一个两个句子拼接的处理')
    args = parser.parse_args()

    # 2，不同进程设置不同的随机种子
    # 设置 torch 与 numpy 的随机种子
    torch.manual_seed(1234 + local_rank * 10)
    np.random.seed(1234 + local_rank * 10)

    # 3，使得批次可以整除显卡个数，然后把全局批次大小（如64）改为单卡批次大小（若2卡，则是32）
    if world_size > 1:
        assert args.batch_size % torch.cuda.device_count() == 0
        args.batch_size = args.batch_size // torch.cuda.device_count()
        logging.info(f"每个 GPU 上执行的批次数是：{args.batch_size}")
        # 将进程号和GPU号对应起来
        torch.cuda.set_device(local_rank)
        # nccl 是 NVIDIA的集合通信库，专为GPU间通信优化
        dist.init_process_group(backend="nccl")
        # 方便用于后面的 .to(device)
        device = torch.device('cuda:{}'.format(local_rank))

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
    # 如果句子过长，仅保留 512 个 token，被截断的一侧是：左边的，即保留文本后半段
    tokenizer.truncation_side = "left"
    # 用户没有给出验证集，则从测试集中划分出
    if df_valid is None:
        sentences = sentence_process(args, df)
        labels = df['label'].tolist()
        encoded_inputs = tokenizer(
            sentences,
            padding=True,
            # 启动截断
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
    train_sample = TensorDataset(train_data['input_ids'], train_data['attention_mask'], train_data['labels'])
    val_sample = TensorDataset(val_data['input_ids'], val_data['attention_mask'], val_data['labels'])
    if world_size >1:
        # shuffle=True 不能写在 DataLoader 中，应该放在 DistributedSampler 中。
        # DistributedSampler 非常重要，它负责把全局批次（如64）随机拆分成 N 份，并保证 **每一份（每一个GPU）上面的样本不重复不缺少**
        # 分布式中不可以写 train_dataloader = DataLoader(train_sample, batch_size=args.batch_size, shuffle=True)，因为这会使每个每张显卡都训练全局批次，等于重复训练了，一批数据被训练 N 次
        train_sampler = DistributedSampler(train_sample,shuffle=True)
        train_dataloader = DataLoader(train_sample, sampler=train_sampler, batch_size=args.batch_size)
        val_sampler = DistributedSampler(val_sample, shuffle=True)
        val_dataloader = DataLoader(val_sample, sampler=train_sampler, batch_size=args.batch_size)
    else:
        train_dataloader = DataLoader(train_sample, batch_size=args.batch_size, shuffle=True)
        val_dataloader = DataLoader(val_sample, batch_size=args.batch_size, shuffle=True)
    model = BertForSequenceClassification.from_pretrained(
        "/Users/nowcoder/workspace/bert_classification/chinese-bert-wwm", num_labels=2).to(device)

    if torch.cuda.device_count() > 1:
        print(f"有 {torch.cuda.device_count()} 个GPU，使用 DDP")
        # model = nn.DataParallel(model)
        model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

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