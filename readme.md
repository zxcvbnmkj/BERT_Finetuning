# 微调 BERT 模型用于二分类任务

## Start

```
conda create -n bert python=3.10
conda activate bert
pdm init
```

可选：查看当前环境中存在的包。使用 pdm 与 pip 得到的结果是一样的

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

## 文件说明

- `main.py` : 训练代码
- `test_code.py` : 在有标签的测试集上测试模型性能
- `infer.py` : 在无标签数据上进行推理并保存正负样本的预测概率和最终预测结果

## 新增改进

- [ ] 利用 PDM 管理环境，做到一键安装
- [ ] 兼容 csv 、json 多种格式数据
- [ ] 推理时仅需加载微调后模型

## 待改进

1. 存储微调后模型时，使用 `model.save_pretrained()` 函数存储微调模型，此时除了 `BertTokenizer` 以外的其余部分都会被存储，文件大小与完整 `BERT` 文件差不多。接下来需要研究是否在微调部分能避免存储整个 `BERT`；或者把 `BertTokenizer` 部分也存在微调模型里，这样就不需要加载 `预训练 BERT`。


* **安全漏洞** ：PyTorch 2.6 之前版本存在严重安全漏洞（CVE-2025-32434）
* **版本要求** ：Hugging Face Transformers 现在要求 PyTorch 至少 2.6 版本才能使用** **`torch.load`

```
Traceback (most recent call last):
  File "/Users/nowcoder/workspace/bert_classification/predict2.py", line 36, in <module>
    model = BertForSequenceClassification.from_pretrained(f'{dir_name}/bert_classifier')
  File "/opt/miniforge3/envs/test/lib/python3.10/site-packages/transformers/modeling_utils.py", line 277, in _wrapper
    return func(*args, **kwargs)
  File "/opt/miniforge3/envs/test/lib/python3.10/site-packages/transformers/modeling_utils.py", line 5048, in from_pretrained
    ) = cls._load_pretrained_model(
  File "/opt/miniforge3/envs/test/lib/python3.10/site-packages/transformers/modeling_utils.py", line 5316, in _load_pretrained_model
    load_state_dict(checkpoint_files[0], map_location="meta", weights_only=weights_only).keys()
  File "/opt/miniforge3/envs/test/lib/python3.10/site-packages/transformers/modeling_utils.py", line 508, in load_state_dict
    check_torch_load_is_safe()
  File "/opt/miniforge3/envs/test/lib/python3.10/site-packages/transformers/utils/import_utils.py", line 1647, in check_torch_load_is_safe
    raise ValueError(
ValueError: Due to a serious vulnerability issue in `torch.load`, even with `weights_only=True`, we now require users to upgrade torch to at least v2.6 in order to use the function. This version restriction does not apply when loading files with safetensors.
See the vulnerability report here https://nvd.nist.gov/vuln/detail/CVE-2025-32434
```
