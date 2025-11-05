import glob
from os import path as osp
import pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from transformers import BertTokenizer
from transformers import BertForSequenceClassification
from utils import eval_classification, concatenate_and_trim_token, calculate_metrics, set_logger

dir_name = osp.dirname(__file__)

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")



if __name__ == '__main__':
    batch_size = 64
    threshold = 0.5
    test_files = glob.glob(osp.join(f"{dir_name}/data", "testset.*"))
    if test_files:
        # 取第一个匹配的文件
        test_file = test_files[0]
        file_ext = osp.splitext(test_file)[1].lower()
        if file_ext == '.csv':
            df = pd.read_csv(test_file)
        elif file_ext == '.json':
            df = pd.read_json(test_file)
        else:
            df = None
            raise ValueError(f"不支持的文件格式: {file_ext}，仅支持 .json 或 .csv")
    else:
        raise FileNotFoundError(f"data 文件夹下没有 testset 文件")
    print(f"测试集大小是：{len(df)}")
    # sentences = df['text'].tolist()
    sentences = df.apply(concatenate_and_trim_token, axis=1).tolist()
    labels = df['label'].tolist()
    tokenizer = BertTokenizer.from_pretrained(f'{dir_name}/dp1_adamw_best_2_bert_classifier')
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
    model = BertForSequenceClassification.from_pretrained(f'{dir_name}/dp1_adamw_best_2_bert_classifier')
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)
    total_acc, total_p, total_r, total_f1 = 0, 0, 0, 0
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
        pred = nn.functional.softmax(torch.tensor(logits), dim=-1).numpy()
        pred = (pred[:, 1] > threshold).astype(int)
        predictions.extend(pred)
        true_labels.extend(label_ids)
        acc, p, r, f1 = calculate_metrics(logits, label_ids, threshold)
        total_acc += acc
        total_p += p
        total_r += r
        total_f1 += f1
    avg_acc = total_acc / len(prediction_dataloader)
    avg_p = total_p / len(prediction_dataloader)
    avg_r = total_r / len(prediction_dataloader)
    avg_f1 = total_f1 / len(prediction_dataloader)
    print(f"Test Metrics: {avg_acc:.4f}, {avg_p: 4f}, {avg_r:4f},{avg_f1:4f}")
    eval_classification(pd.Series(true_labels), pd.Series(predictions), "测试集")