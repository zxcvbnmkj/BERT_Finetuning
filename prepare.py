"""
用于把数据转换为：训练集、测试集（验证集可选）形式
"""
import glob
import json
import os

import pandas as pd


def data_transform(dir_name):
    print("数据预处理中...")
    data_list = []
    json_files = glob.glob(os.path.join(f"{dir_name}/json_files", "*.json"))
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
    test_set.to_json(f'{dir_name}/data/testset.json', orient='records', force_ascii=False)
    train_set.to_json(f'{dir_name}/data/trainset.json', orient='records', force_ascii=False)
    return train_set
if __name__ == '__main__':
    df = data_transform()