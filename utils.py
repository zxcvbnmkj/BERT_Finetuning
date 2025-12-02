import os
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import logging

def set_logger(log_name):
    logging.basicConfig(
        filename=f"{log_name}.log",
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        encoding='utf-8'
    )

def load_files(train_file, valid_file):
    df_valid = None
    _, ext = os.path.splitext(train_file)
    ext = ext.lower()
    if ext == '.json':
        df_train = pd.read_json(train_file)
        if valid_file is not None:
            df_valid = pd.read_json(valid_file)
    elif ext == '.csv':
        df_train = pd.read_csv(train_file)
        if valid_file is not None:
            df_valid = pd.read_csv(valid_file)
    elif ext == '.xlsx':
        df_train = pd.read_excel(train_file)
        if valid_file is not None:
            df_valid = pd.read_excel(valid_file)
    else:
        raise ValueError(f'当前使用的文件后缀 {ext} 不被支持，仅支持 josn、csv、excel')
    return df_train, df_valid


def eval_classification(y_true: pd.Series, y_pred: pd.Series, title="无", use_log=False):
    # acc P R F1 support 等指标的报告
    # accuracy = report['accuracy']
    # macro_precision = report['macro avg']['precision']
    # macro_recall = report['macro avg']['recall']
    # macro_f1 = report['macro avg']['f1-score']
    # class_0_precision = report['0']['precision']
    # class_0_recall = report['0']['recall']
    report = classification_report(y_true, y_pred, output_dict=True)
    metrics_df = pd.DataFrame(report).transpose()
    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    classes = metrics_df.index.to_numpy()[:-3]
    cm_df = pd.DataFrame(cm, index=classes, columns=classes)
    if use_log:
        logging.info(f"\nMetrics by Class: ({title})")
        logging.info(metrics_df)
        logging.info("Confusion Matrix:")
        logging.info(cm_df)
    print(f"\nMetrics by Class: ({title})")
    print(metrics_df)
    print("Confusion Matrix:")
    print(cm_df)
    return report

# 不必手动拼接
# def sentence_process(args, df):
#     if args.task == 0:
#         sentences = df['answer'].apply(lambda x: str(x)[-1200:]).tolist()
#     elif args.task == 1:
#         sentences = df.apply(concatenate_and_trim_token, axis=1).tolist()
#     else:
#         raise ValueError(f"不支持的任务类型: {args.task}，支持 0 或 1")
#     return sentences
#
#
# def concatenate_and_trim_token(row):
#     combined_text = str(row['answer']) + '[SEP]' + str(row['question'])
#     if len(combined_text) > 1200:
#         return combined_text[-1200:]
#     else:
#         return combined_text

def tokenizering(task, tokenizer, df):
    if task == 1:
        df['question'] = df['question'].apply(
            lambda x: str(x)[-200:]
        )
        encoded_inputs = tokenizer(
            df['question'].tolist(),
            df['answer'].tolist(),
            padding=True,
            truncation='only_second',
            max_length=512,
            return_tensors='pt')
    elif task == 0:
        encoded_inputs = tokenizer(
            df['answer'].tolist(),
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
    return encoded_inputs