## 待改进
1. 存储微调后模型时，使用 `model.save_pretrained()` 函数存储微调模型，此时除了 `BertTokenizer` 以外的其余部分都会被存储，文件大小与完整 `BERT` 文件差不多。接下来需要研究是否在微调部分能避免存储整个 `BERT`；或者把`BertTokenizer` 部分也存在微调模型里，这样就不需要加载 `预训练 BERT`。 