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
    batch_size = 80
    df = pd.read_json(
        './ans1104.json')
    print("预测样本个数是", len(df))
    sentences = df['answer'].tolist()
    tokenizer = BertTokenizer.from_pretrained('./written3_10_bert_classifier')
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
    model = BertForSequenceClassification.from_pretrained(f'{dir_name}/written3_10_bert_classifier')
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
        probabilities_list0.extend(probabilities[:, 0])
        probabilities_list1.extend(probabilities[:, 1])
        pred_indices = np.argmax(logits, axis=1)
        predictions.extend(np.where(pred_indices == 1, '书面语', '非书面语'))
    result_df = pd.DataFrame({
        'id': df['id'],  # 保留原第0列作为ID
        'question': df['question'],
        'answer': df['answer'],  # 保留原第9列作为text
        '非书面语概率': probabilities_list0,  # 概率列表
        '书面语概率': probabilities_list1,
        'predict': predictions  # 预测结果
    })
    result_df.to_json("written_lang_predict.json", orient='records', force_ascii=False)