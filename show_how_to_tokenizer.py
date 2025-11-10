from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained(
    f'/System/Volumes/Data/data/models/check_completion_v2/20250731170901/chinese-bert-wwm')

# 设置截断后舍弃左边，保留右边
tokenizer.truncation_side = "left"

question = "请描述一次你在学习或项目中遇到的重大变化。说说你是如何调整自己来适应这个变化的？你从中学到了什么？"
answer = "好的，我将针对以上问题进行回答。我想说的是，在实习的。过程当中。有一次外出采访与拍摄。本来是要拍摄外景的，但是突然下了大暴雨，那么对于这个重大变化来说呢？首先，我们出勤的时候是查了天气预报的。因为要出外勤，所以我们肯定要对当时的环境啊进行一个相关的了解。我们也是很及时的就把相关的工具准备好了。我们把伞带好了之后呢。我给我们的拍摄老师。把这个防护做好之后，我们也是攻克了这个难关，那么同时呢，我们也会采取外景转内景的这么一个方法。我们对，因为外景它是自然光，也许呈现的视频效果不会特别好。所以我们会有第二个计划，我们会安排一个内景，内景的时候呢，它会有打光，比如说呃侧光啊，它显得人物更立体呀，更丰富，这样来呃适应这个变化，我们每次出行首先做好呃完整的基础的一。一个准备，那么其次呢，我们会呃制定这个第二个计划，那么第二个计划也不一定会比第一个差。所以来做到一个工作的协调。以上是我的回答。"

test_data = tokenizer(question, answer, max_length=512,
                      truncation='only_second')  # 只截断第二个句子
print(test_data['input_ids'])
print(test_data['token_type_ids'])
print(len(test_data['input_ids']))

# 将 input_ids 转换回 tokens，方便查看分词结果
tokens = tokenizer.convert_ids_to_tokens(test_data['input_ids'])
print("Tokens:", tokens)
print("Token数量:", len(tokens))
