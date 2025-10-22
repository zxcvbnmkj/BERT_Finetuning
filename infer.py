from os import path as osp
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from transformers import BertTokenizer
from transformers import BertForSequenceClassification

dir_name = osp.dirname(__file__)

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

if __name__ == '__main__':
    batch_size = 64
    df = pd.read_excel(
        './data.xlsx')
    # 数据无列名，只能从列编号取值
    sentences = df.iloc[:, 9].tolist()
    tokenizer = BertTokenizer.from_pretrained('bert_classifier')
    test_data = tokenizer(
        sentences,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors='pt'
    )
    prediction_dataset = TensorDataset(test_data['input_ids'], test_data['attention_mask'])
    prediction_dataloader = DataLoader(prediction_dataset, sampler=SequentialSampler(prediction_dataset),
                                       batch_size=batch_size)
    model = BertForSequenceClassification.from_pretrained(f'{dir_name}/bert_classifier')
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)
    predictions = []
    probabilities_list0 = []
    probabilities_list1 = []
    model.eval()
    for batch in prediction_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask = batch
        with torch.no_grad():
            outputs = model(b_input_ids, attention_mask=b_input_mask)
        logits = outputs.logits.detach().cpu().numpy()
        probabilities = torch.softmax(torch.tensor(logits), dim=1).numpy()
        probabilities_list0.extend(probabilities[:,0])
        probabilities_list1.extend(probabilities[:,1])
        predictions.extend(np.argmax(logits, axis=1))
    result_df = pd.DataFrame({
         # 第 0 列是数据的 ID 列
        'ID': df.iloc[:, 0],
        'text': df.iloc[:, 9],
        '非书面语概率': probabilities_list0,
        '书面语概率':probabilities_list1,
        '预测结果': predictions
    })
    result_df.to_excel("预测结果.xlsx", index=False)
