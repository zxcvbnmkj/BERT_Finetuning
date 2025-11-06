import pandas as pd

df=pd.read_json("./data/trainset.json")

question_char_len = df['question'].str.len()
answer_char_len = df['answer'].str.len()

print("=== 字符长度统计 ===")
print(f"问题长度范围: {question_char_len.min()} ~ {question_char_len.max()}")
print(f"回答长度范围: {answer_char_len.min()} ~ {answer_char_len.max()}")
print(f"问题平均长度: {question_char_len.mean():.1f}")
print(f"回答平均长度: {answer_char_len.mean():.1f}")
print(f"问题长度大于 512 的样本个数：", len(df[question_char_len > 512]))
print(f"问题长度大于 200 的样本个数：", len(df[question_char_len > 200]))


label_counts = df['label'].value_counts()
print(f"\n=== 标签分布 ===")
print(f"label为1的样本个数: {label_counts.get(1, 0)}")
print(f"label为0的样本个数: {label_counts.get(0, 0)}")