"""
测试是否可正常执行：torchrun --nproc_per_node=2 train_DDP.py --sub_num 40
"""
# -*- coding: utf-8 -*-
import argparse
import glob
import logging
import os
import warnings
import torch
import torch.nn as nn
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

from loss import focal_loss
from model import BertCustomClassification
from utils import eval_classification, set_logger, data_transform, tokenizering, load_files

warnings.filterwarnings("ignore")

local_rank = int(os.environ.get("LOCAL_RANK", 0))
# 进程数，等于使用到了 GPU 数目
world_size = int(os.environ.get("WORLD_SIZE", 1))
# 当前是否主进程
is_main_process = (local_rank <= 0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dir_name = osp.dirname(__file__)


def finetuning(epochs, max_patient, threshold, model_type, type):
    best_value = 0
    patient = 0
    for epoch_i in trange(epochs, desc="Epoch"):
        # ========== 训练阶段 ==========
        if world_size > 1:
            train_sampler.set_epoch(epoch_i)
        if is_main_process:
            logging.info(f"当前是第{epoch_i}轮")
            logging.info("================训练中==================")
        total_train_loss, train_true_labels, train_predictions, val_true_labels, val_predictions = 0, [], [], [], []
        for index, batch in enumerate(train_dataloader):
            b_input_ids, b_input_mask, b_type, b_labels = tuple(t.to(device) for t in batch)
            model.zero_grad()
            outputs = model(input_ids=b_input_ids,
                            attention_mask=b_input_mask,
                            token_type_ids=b_type,
                            labels=b_labels)
            if model_type == 'official':
                loss = outputs.loss
                logits = outputs.logits.detach().cpu().numpy()
            elif model_type == 'custom':
                loss = loss_func(outputs, torch.tensor(b_labels))
                logits = outputs.detach().cpu().numpy()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            label_ids = b_labels.to('cpu').numpy()
            total_train_loss += loss.item()
            # report 形式的指标
            train_true_labels.extend(label_ids)
            pred = nn.functional.softmax(torch.tensor(logits), dim=-1).numpy()
            pred = (pred[:, 1] > threshold).astype(int)
            train_predictions.extend(pred)
        avg_train_loss = total_train_loss / len(train_dataloader)
        # 同步所有进程的指标 ，若不加则只能得到 主进程上面的批次数量数据
        if world_size > 1:
            # 将列表转换为张量
            true_labels_tensor = torch.tensor(train_true_labels, device=device)
            predictions_tensor = torch.tensor(train_predictions, device=device)
            # 创建指定形状的值为 0 的张量，形状是 单卡 tensor 形状 * 卡数
            gathered_true_labels = [torch.zeros_like(true_labels_tensor) for _ in range(world_size)]
            gathered_predictions = [torch.zeros_like(predictions_tensor) for _ in range(world_size)]
            # 收集所有进程的数据到主进程，
            # 使得每个GPU的 gathered_true_labels 列表都会包含所有GPU的数据
            dist.all_gather(gathered_true_labels, true_labels_tensor)
            dist.all_gather(gathered_predictions, predictions_tensor)
            if is_main_process:
                all_true_labels_global = []
                all_predictions_global = []
                for i in range(world_size):
                    # 遍历所有GPU收集的数据，合并成全局列表。gathered_true_labels[i] 是第 i 个卡指标数据
                    all_true_labels_global.extend(gathered_true_labels[i].cpu().numpy())
                    all_predictions_global.extend(gathered_predictions[i].cpu().numpy())
                logging.info(f"\nEpoch {epoch_i + 1}/{epochs}")
                logging.info(f"Train loss: {avg_train_loss:.4f}")
                eval_classification(pd.Series(all_true_labels_global), pd.Series(all_predictions_global),
                                    f"训练集_轮{epoch_i}",
                                    use_log=True)
        else:
            eval_classification(pd.Series(train_true_labels), pd.Series(train_predictions), f"训练集_轮{epoch_i}",
                                use_log=True)

        if is_main_process:
            logging.info("==============验证中===============")
        model.eval()
        with torch.no_grad():
            for batch in val_dataloader:
                b_input_ids, b_input_mask, b_type, b_labels = tuple(t.to(device) for t in batch)
                outputs = model(input_ids=b_input_ids,
                                attention_mask=b_input_mask,
                                token_type_ids=b_type)
                if model_type == 'official':
                    loss = outputs.loss
                    logits = outputs.logits.detach().cpu().numpy()
                elif model_type == 'custom':
                    loss = loss_func(outputs, torch.tensor(b_labels))
                    logits = outputs.detach().cpu().numpy()
                label_ids = b_labels.to('cpu').numpy()
                # report 形式的指标
                val_true_labels.extend(label_ids)
                pred = nn.functional.softmax(torch.tensor(logits), dim=-1).numpy()
                pred = (pred[:, 1] > threshold).astype(int)
                val_predictions.extend(pred)
        if world_size > 1:
            # 将列表转换为张量
            val_true_labels_tensor = torch.tensor(val_true_labels, device=device)
            val_predictions_tensor = torch.tensor(val_predictions, device=device)
            # 收集所有进程的数据到主进程，"循环 world_size 次"
            val_gathered_true_labels = [torch.zeros_like(val_true_labels_tensor) for _ in range(world_size)]
            val_gathered_predictions = [torch.zeros_like(val_predictions_tensor) for _ in range(world_size)]
            # 使得每个GPU的 gathered_true_labels 列表都会包含所有GPU的数据
            dist.all_gather(val_gathered_true_labels, val_true_labels_tensor)
            dist.all_gather(val_gathered_predictions, val_predictions_tensor)
            if is_main_process:
                val_all_true_labels_global = []
                val_all_predictions_global = []
                for i in range(world_size):
                    # 遍历所有GPU收集的数据，合并成全局列表
                    val_all_true_labels_global.extend(val_gathered_true_labels[i].cpu().numpy())
                    val_all_predictions_global.extend(val_gathered_predictions[i].cpu().numpy())
                report = eval_classification(pd.Series(val_all_true_labels_global),
                                             pd.Series(val_all_predictions_global),
                                             f"验证集_轮{epoch_i}", use_log=True)
        else:
            report = eval_classification(pd.Series(val_true_labels), pd.Series(val_predictions), f"验证集_轮{epoch_i}",
                                         use_log=True)
        if is_main_process:
            # epoch_value = (report['0']['recall'] + report['1']['precision']) / 2.0
            epoch_value = report['1']['f1-score']
            for name, param in model.named_parameters():
                if param is not None:
                    param.data = param.data.contiguous()
            if epoch_value > best_value:
                logging.info(
                    f"存储当前最佳模型，属于轮{epoch_i}，它的指标为{epoch_value}，模型存储在{dir_name}/{type}_best_{epoch_i}_bert_classifier")
                best_value = epoch_value
                patient = 0
                # 如果使用了分布式训练
                if hasattr(model, 'module'):
                    model.module.save_pretrained(f'{dir_name}/{type}_best_{epoch_i}_bert_classifier')
                else:
                    model.save_pretrained(f'{dir_name}/{type}_best_{epoch_i}_bert_classifier')
                tokenizer.save_pretrained(f'{dir_name}/{type}_best_{epoch_i}_bert_classifier')
            else:
                patient += 1
                logging.info(
                    f"存储当前非最佳模型，属于轮{epoch_i}，它的指标为{epoch_value}，模型存储在{dir_name}/{type}_{epoch_i}_bert_classifier")
                if hasattr(model, 'module'):
                    model.module.save_pretrained(f'{dir_name}/{type}_{epoch_i}_bert_classifier')
                else:
                    model.save_pretrained(f'{dir_name}/{type}_{epoch_i}_bert_classifier')
                tokenizer.save_pretrained(f'{dir_name}/{type}_{epoch_i}_bert_classifier')
        # 同步patient给所有进程
        if world_size > 1:
            patient_tensor = torch.tensor(patient, device=device)
            dist.all_reduce(patient_tensor)
            patient = patient_tensor.item()  # 所有进程得到相同的patient值
        if patient == max_patient:
            break

    # 清理分布式资源
    if world_size > 1:
        dist.destroy_process_group()
    logging.info("程序运行完毕，正常退出！")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--train_file', type=str, default='./data/trainset.json')
    parser.add_argument('--valid_file', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=80)
    parser.add_argument('--max_patient', type=int, default=3, help='最大容忍次数')
    parser.add_argument('--threshold', type=float, default=0.5, help='认定为正类（1）的阈值')
    parser.add_argument('--sub_num', type=int, default=0, help='如果不为 0 表示使用子数据集训练与验证，将从原数据集中截取 sub_num 条数据')
    parser.add_argument('--task', type=int, default=0, help='0: 单句任务；'
                                                            '1: 双句任务，增加一个两个句子拼接的处理')
    parser.add_argument('--log_name', type=str, default='anwser_only', help='日志文件名字')
    parser.add_argument('--model', type=str, default='custom', help='模型形式：custom（自定义结构）、official（Transform库提供的官方结构）')
    args = parser.parse_args()
    set_logger(args.log_name)

    # 2，不同进程设置不同的随机种子
    # 设置 torch 与 numpy 的随机种子
    torch.manual_seed(1234 + local_rank * 10)
    np.random.seed(1234 + local_rank * 10)

    # 3，使得批次可以整除显卡个数，然后把全局批次大小（如64）改为单卡批次大小（若2卡，则是32）
    if world_size > 1:
        assert args.batch_size % torch.cuda.device_count() == 0
        args.batch_size = args.batch_size // torch.cuda.device_count()
        logging.info(f"每个 GPU 上执行的批次数是：{args.batch_size}")
        # nccl 是 NVIDIA的集合通信库，专为GPU间通信优化
        dist.init_process_group(backend="nccl")
        # 将进程号和GPU号对应起来
        torch.cuda.set_device(local_rank)
        # 方便用于后面的 .to(device)
        device = torch.device('cuda:{}'.format(local_rank))

    df, df_valid = load_files(args.train_file, args.valid_file)

    # 仅取 sub_num 条测试代码是否正确
    # df = df.head(args.sub_num)
    if args.sub_num != 0:
        df_0 = df[df['label'] == 0].head(int(args.sub_num / 2.0))
        df_1 = df[df['label'] == 1].head(int(args.sub_num / 2.0))
        df = pd.concat([df_0, df_1]).sample(frac=1).reset_index(drop=True)
        print("子数据集样本数：", len(df))
    tokenizer = BertTokenizer.from_pretrained(f'{dir_name}/chinese-bert-wwm')
    # 如果句子过长，仅保留 512 个 token，被截断的一侧是：左边的，即保留文本后半段
    # tokenizer.truncation_side = "left"
    encoded_inputs = tokenizering(args.task, tokenizer, df)
    # 用户没有给出验证集，则从测试集中划分出
    if df_valid is None:
        labels = df['label'].tolist()
        train_inputs, val_inputs, train_masks, val_masks, train_type, val_type, train_labels, val_labels = train_test_split(
            encoded_inputs['input_ids'],
            encoded_inputs['attention_mask'],
            encoded_inputs['token_type_ids'],
            labels,
            test_size=0.1,
            random_state=42
        )
        train_sample = TensorDataset(train_inputs, train_masks, train_type, torch.tensor(train_labels))
        val_sample = TensorDataset(val_inputs, val_masks, val_type, torch.tensor(val_labels))
    else:
        train_labels = df['label'].tolist()
        train_inputs = encoded_inputs['input_ids']
        train_masks = encoded_inputs['attention_mask']
        train_type = encoded_inputs['token_type_ids']
        if args.sub_num != 0:
            df_valid = df_valid.sample(n=20, random_state=42)
        logging.info("验证集长度：" + str(len(df_valid)))
        val_labels = df_valid['label'].tolist()
        encoded_inputs_valid = tokenizering(args.task, tokenizer, df_valid)
        train_sample = TensorDataset(encoded_inputs['input_ids'], encoded_inputs['attention_mask'],
                                     encoded_inputs['token_type_ids'], torch.tensor(train_labels))
        val_sample = TensorDataset(encoded_inputs_valid['input_ids'], encoded_inputs_valid['attention_mask'],
                                   encoded_inputs_valid['token_type_ids'], torch.tensor(val_labels))

    if world_size > 1:
        # shuffle=True 不能写在 DataLoader 中，应该放在 DistributedSampler 中。
        # DistributedSampler 非常重要，它负责把全局批次（如64）随机拆分成 N 份，并保证 **每一份（每一个GPU）上面的样本不重复不缺少**
        # 分布式中不可以写 train_dataloader = DataLoader(train_sample, batch_size=args.batch_size, shuffle=True)，因为这会使每个每张显卡都训练全局批次，等于重复训练了，一批数据被训练 N 次
        train_sampler = DistributedSampler(train_sample, shuffle=True)
        train_dataloader = DataLoader(train_sample, sampler=train_sampler, batch_size=args.batch_size)
        val_sampler = DistributedSampler(val_sample, shuffle=True)
        val_dataloader = DataLoader(val_sample, sampler=val_sampler, batch_size=args.batch_size)
    else:
        train_dataloader = DataLoader(train_sample, batch_size=args.batch_size, shuffle=True)
        val_dataloader = DataLoader(val_sample, batch_size=args.batch_size, shuffle=True)

    if args.model == 'official':
        model = BertForSequenceClassification.from_pretrained(
            f'{dir_name}/chinese-bert-wwm', num_labels=2, classifier_dropout=0.5).to(device)
    elif args.model == 'custom':
        model = BertCustomClassification(
            f'{dir_name}/chinese-bert-wwm').to(device)
        loss_func = focal_loss(alpha=0.1)

    if torch.cuda.device_count() > 1:
        if is_main_process:
            print(f"有 {torch.cuda.device_count()} 个GPU，使用 DDP")
        model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank,
                                        find_unused_parameters=True)

    optimizer = AdamW(model.parameters(), lr=1e-5)

    finetuning(epochs=args.epochs, max_patient=args.max_patient, threshold=args.threshold, model_type=args.model,
               type=args.log_name)
