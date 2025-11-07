"""
前往注意要根据任务修改截断方向：tokenizer.truncation_side = "left"
"""
import glob
from os import path as osp
import pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from transformers import BertTokenizer
from transformers import BertForSequenceClassification
from utils import eval_classification

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
    # sentences = df.apply(concatenate_and_trim_token, axis=1).tolist()
    df['question'] = df['question'].apply(
        lambda x: str(x)[-200:]
    )
    labels = df['label'].tolist()
    tokenizer = BertTokenizer.from_pretrained(f'{dir_name}/dp1_adamw_best_2_bert_classifier')
    tokenizer.truncation_side = "left"
    test_data = tokenizer(
        df['question'].tolist(),
        df['answer'].tolist(),
        padding=True,
        truncation='only_second',
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
    eval_classification(pd.Series(true_labels), pd.Series(predictions), "测试集")