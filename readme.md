# 微调 BERT 模型用于二分类任务

## Start

给 PDM 配置镜像源

```
pdm config pypi.url https://mirrors.aliyun.com/pypi/simple/
```

激活虚拟环境

```
source .venv/bin/activate
```

也可以不激活虚拟环境，只有使用 PDM 运行

```
   pdm run python main.py
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
