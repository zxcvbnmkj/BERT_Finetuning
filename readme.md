# 微调 BERT 模型用于二分类任务
## Start
准备环境，并使用 PDM 初始化项目
```
conda create -n bert python=3.10
conda activate bert
pdm init
```
可选：查看当前环境中存在的包。使用 pdm 与 pip 得到的结果是一样的。说明二者安装区间是共享的
```
(bert)  🐍 bert  ~/BERT_Finetuning   master ±  pdm list
╭────────────┬─────────┬───────────────────────────────────────────────────────────────────╮
│ name       │ version │ location                                                          │
├────────────┼─────────┼───────────────────────────────────────────────────────────────────┤
│ pip        │ 25.2    │ /home/conda/feedstock_root/build_artifacts/pip_1753924886980/work │
│ setuptools │ 80.9.0  │                                                                   │
│ wheel      │ 0.45.1  │                                                                   │
╰────────────┴─────────┴───────────────────────────────────────────────────────────────────╯
```
```
(bert)  🐍 bert  ~/BERT_Finetuning   master ±  pip list               
Package    Version
---------- -------
pip        25.2
setuptools 80.9.0
wheel      0.45.1
```
使用 pip 下载 pytorch 的 CPU 版本，PDM 会下载 gpu 版本的。因为每个设备不同，所以 torch 没必要使用 `pdm.lock` 来同步
```
pip install torch
```
给 PDM 配置镜像源
```
pdm config pypi.url https://mirrors.aliyun.com/pypi/simple/
```
通过 PDM 安装其余依赖
```
pdm sync
```
由于本项目基于 intel CPU 的 MAC 系统编写，受限于操作系统最高仅可以使用 `pytorch 2.2.2`，为了和它配套，安装了 `transformers==4.43.0` 和 `numpy==1.24.3` 它们都属于旧版本
## 文件说明
- `data` 文件夹下面存放 训练、验证、测试集，分别命名为 `trainset` 、`validset`、`testset `，可以是 `csv `与 `json` 两种格式
- `main.py` : 训练代码
- `test_code.py` : 在有标签的测试集上测试模型性能
- `infer.py` : 在无标签数据上进行推理并保存正负样本的预测概率和最终预测结果
## 新增改进
- [X] 利用 PDM 管理环境，做到一键安装环境
- [X] 兼容 csv 、json 多种格式数据
- [X] 推理时仅需加载微调后模型
## 待改进
1. 【已完成】存储微调后模型时，使用 `model.save_pretrained()` 函数存储微调模型，此时除了 `BertTokenizer` 以外的其余部分都会被存储，文件大小与完整 `BERT` 文件差不多。接下来需要研究是否在微调部分能避免存储整个 `BERT`；或者把 `BertTokenizer` 部分也存在微调模型里，这样就不需要加载 `预训练 BERT`。
## 关于存储方法
1. save_pretrained 只会保持模型本身，不保存分词器。若需要分词器，还要额外保存 `tokenizer.save_pretrained`
但其实微调是不会改变分词器的
```
if hasattr(model, 'module'):
    model.module.save_pretrained(f'{dir_name}/bert_classifier')
else:
    model.save_pretrained(f'{dir_name}/bert_classifier')
tokenizer.save_pretrained(f'{dir_name}/bert_classifier')
```
微调后模型本身只会有：`model.safetensors` 和 `config.json` 两个文件
如果把分词器也保存的话则有：`special_tokens_map.json`、`tokenizer_config.json`、`vocab.txt` 三个文件
## 关于模型加载器的两种写法
使用 `RandomSampler` 打乱数据集样本
```
train_sample = TensorDataset(train_data['input_ids'], train_data['attention_mask'], train_data['labels'])
# 使用了 RandomSampler 包裹数据类，使得每一轮都会打乱其中的样本
# 但是这种写法并没有直接在 DataLoader 里面使用 shuffle 那么简单
train_sampler = RandomSampler(train_sample)
train_dataloader = DataLoader(train_sample, sampler=train_sampler, batch_size=args.batch_size)
```
`sampler` 参数和 `shuffle` 参数是互斥的，只能存在一个
```
train_sample = TensorDataset(train_data['input_ids'], train_data['attention_mask'], train_data['labels'])
train_dataloader = DataLoader(train_sample, batch_size=args.batch_size, shuffle=True)
```
## 关于一个警告
```
Some weights of BertForSequenceClassification were not initialized from the model checkpoint at /Users/nowcoder/workspace/bert_classification/chinese-bert-wwm and are newly initialized: ['classifier.bias', 'classifier.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
意思是：
BertForSequenceClassification 的一些权重没有从模型检查点 /Users/nowcoder/workspace/bert_classification/chinese-bert-wwm 初始化，而是被新初始化的：['classifier.bias', 'classifier.weight']
你可能需要在下游任务上训练这个模型，才能将其用于预测和推理。
```
因为 BertForSequenceClassification 在 BERT 的基础上新加了一个层，这些层不能从预训练BERT 中获取权重，所以必须微调

## 关于 deepspeed 的配置项
```
{
  "train_batch_size": 128,   # 全局批次大小
  "gradient_accumulation_steps": 4,  # // 梯度累积步数
  "optimizer": {
    "type": "Adam",
    "params": {
      "lr": 0.00015,
      "eps": 1e-8    # 数值稳定性参数
    }
  },
  "scheduler": {
    "type": "WarmupDecayLR",   // DeepSpeed 自定义的一种学习率调度器，它结合了 热身（Warmup） 和 衰减（Decay） 两个阶段
    "params": {
      "warmup_min_lr": 0,    // 热身起始学习率
      "warmup_max_lr": 1.5e-4,     // 热身结束学习率  
      "warmup_num_steps": 1000,     // 热身步数
      "warmup_type": "cosine" // 热身阶段学习率以 consine形式变化
    }
  },
  "fp16": {
    "enabled": true,   // 启用FP16训练，即混合精度学习
    "loss_scale": 0,   // 0=动态loss scaling, 其他值=静态loss scaling
    "loss_scale_window": 1000,    // 动态loss scaling的窗口大小
  },
  "zero_optimization": {
    "stage": 2,
    "contiguous_gradients": true,
    "cpu_offload": true    // 是否启用 CPU 卸载
  },
  "gradient_clipping": 1.0,     // 梯度裁剪阈值
}
```
