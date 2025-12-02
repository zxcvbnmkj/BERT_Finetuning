from torch import nn
from transformers import BertModel, BertTokenizer


class BertCustomClassification(nn.Module):
    def __init__(self, pretrained):
        super(BertCustomClassification, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained)
        self.dropout_pair = nn.Dropout(0.5)
        self.dense = nn.Linear(768, 2)

    def forward(self, input_ids, attention_mask, token_type_ids):
        output = self.bert(input_ids=input_ids, token_type_ids=token_type_ids,
                           attention_mask=attention_mask)
        #  output 其实是一个类，它比较特殊，既是对象又是类字典结构。可以采用类、字典两种方式来访问它
        # <class 'transformers.modeling_outputs.BaseModelOutputWithPoolingAndCrossAttentions'>
        # print(type(output))
        # 获取所有键
        # odict_keys(['last_hidden_state',   形状: (batch_size, seq_len, hidden_size)
        # 'pooler_output'])   形状: (batch_size, hidden_size)， 是 [CLS] token的池化表示
        # print(output.keys())
        # 获取CLS的未池化前的表示，[0]表示隐藏层，[:, 0, :]表示 seq_len = 0 处
        em_sentence = output[0][:, 0, :]
        em_sentence = self.dropout_pair(em_sentence)
        x = self.dense(em_sentence)
        return x


if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained(
        f'/System/Volumes/Data/data/models/check_completion_v2/20250731170901/chinese-bert-wwm')
    tokenizer.truncation_side = "left"
    question = "请描述一次你在学习或项目中遇到的重大变化。说说你是如何调整自己来适应这个变化的？你从中学到了什么？"
    answer = "好的，我将针对以上问题进行回答。我想说的是，在实习的。过程当中。有一次外出采访与拍摄。本来是要拍摄外景的，但是突然下了大暴雨，那么对于这个重大变化来说呢？首先，我们出勤的时候是查了天气预报的。因为要出外勤，所以我们肯定要对当时的环境啊进行一个相关的了解。我们也是很及时的就把相关的工具准备好了。我们把伞带好了之后呢。我给我们的拍摄老师。把这个防护做好之后，我们也是攻克了这个难关，那么同时呢，我们也会采取外景转内景的这么一个方法。我们对，因为外景它是自然光，也许呈现的视频效果不会特别好。所以我们会有第二个计划，我们会安排一个内景，内景的时候呢，它会有打光，比如说呃侧光啊，它显得人物更立体呀，更丰富，这样来呃适应这个变化，我们每次出行首先做好呃完整的基础的一。一个准备，那么其次呢，我们会呃制定这个第二个计划，那么第二个计划也不一定会比第一个差。所以来做到一个工作的协调。以上是我的回答。"
    test_data = tokenizer(question, answer, max_length=512,
                          truncation='only_second', return_tensors='pt')

    bc = BertCustomClassification(
        '/System/Volumes/Data/data/models/check_completion_v2/20250731170901/chinese-bert-wwm')
    preds = bc(test_data['input_ids'], test_data['attention_mask'], test_data['token_type_ids'])
