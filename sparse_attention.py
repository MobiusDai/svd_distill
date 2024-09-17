import math 
import torch
import torch.nn as nn 
from transformers.models.vit.modeling_vit import ViTSelfAttention


class SparseSelfAttention(ViTSelfAttention):
    def __init__(self, config):
        super().__init__(config)
        self.ratio = None 

    def forward(
        self, hidden_states, head_mask, output_attentions: bool = False):
        # print('this is the sparse attention')
        
        mixed_query_layer = self.query(hidden_states)

        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # filter the attention scores by threshold only save the top-k values
        # save top-30% of the values and set the rest to -inf

        k = int(attention_scores.size(-1) * self.ratio)
        k = max(k, 1)  # 确保至少保留一个元素

        # 在最后一个维度上获取前k个最大的注意力得分及其索引
        topk_values, topk_indices = torch.topk(attention_scores, k, dim=-1)

        # 创建一个与attention_scores形状相同的全为负无穷的张量
        mask = torch.full_like(attention_scores, float('-inf'))

        # 将前k个注意力得分填充回对应的位置
        mask.scatter_(-1, topk_indices, topk_values)

        # 更新注意力得分
        attention_scores = mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs