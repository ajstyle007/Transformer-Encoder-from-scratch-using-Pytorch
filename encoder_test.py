import torch
from torch import nn
from encoder_layer import Encoder_block
from positional_encoding import Positional_Encoding

d_model = 512  # main model dimension
num_heads = 8  # number of heads
d_ff = 2048    # feedforward hidden dimension
seq_len = 128  # max input length
vocab_size = 30000

embedding_layer = nn.Embedding(vocab_size, d_model)
pos_encoding = Positional_Encoding(seq_len, d_model)

def prepare_encoder_input(token_ids):
    token_ids = torch.tensor(token_ids).unsqueeze(0)  # (1, seq_len)

    # 1. Convert token IDs → learned embeddings
    x = embedding_layer(token_ids)                      # (1, seq_len, d_model)

    # 2. Add sinusoidal positional encoding
    x = pos_encoding(x)                                 # (1, seq_len, d_model)

    return x

# encoder = Encoder_block(d_model, d_ff, num_heads)

# x = torch.randn(1, 3, d_model)   # batch=1, seq_len=3, embedding size=512

# out = encoder(x)

# print("Input shape: ", x.shape)
# print("Output shape:", out.shape)
# print(out)

# Expected Output
# Input shape:  torch.Size([1, 3, 512])
# Output shape: torch.Size([1, 3, 512])
# tensor([...])

# Encoder stack (N layers)
class Encoder(nn.Module):
    def __init__(self, num_layers, d_model, d_ff, num_heads):
        super().__init__()
        self.layers = nn.ModuleList([
            Encoder_block(d_model, d_ff, num_heads)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
# enc = Encoder(6, 512, 2048, 8)
# x = torch.randn(1, 3, 512)
# out = enc(x)
# print(out.shape)
# print(out)
# torch.Size([1, 3, 512])

# Keyword extraction
tokens = [7, 1542, 98]  # "I love you"
words  = ["I", "love", "you"]

x = prepare_encoder_input(tokens)

encoder_layer = Encoder_block(d_model=512, d_ff=2048, num_heads=8)

output, attn = encoder_layer(x)

attn_avg = attn.mean(dim=0)   # (seq_len, seq_len)
print("attn_avg: ", attn_avg)
attn_avg = attn_avg[0]             # remove batch → (seq, seq)
print("attn_avg: ", attn_avg)

word_importance = attn_avg.mean(dim=0)   # (seq_len,)
print(word_importance)


values, indices = torch.topk(word_importance, k=2)
print("topk: ", values)
indices = indices.tolist()  # [1, 0] (for example)
print("indices: ", indices)

keywords = [words[i] for i in indices]
print("Keywords: ", keywords)
