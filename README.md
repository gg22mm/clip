# clip
改造clip让它支持中文，并支持8K的输入长度

# 已知问题

OpenAI 的 CLIP 也存在三大短板：

1. 文本输入容量非常有限。最多仅支持 77 个 token 的输入，根据 LongCLIP 的实验，实际上其有效输入不超过 20 个 token。

2. 在纯文本检索中表现不佳。主要原因有两点：首先，CLIP 模型的训练目标是对齐文本和图像，没有针对纯文本检索进行专门优化。其次，CLIP 模型的训练数据主要由相对较短的文本组成，难以泛化到更广阔的文本检索场景。

3. 不支持中文

所以想用clip来做向量搜索： 文本-文本、文本-图像、图像-文本、图像-图像四个方向的检索 或 视频 搜索时困难重重。

为什么要用clip来做搜索？ 我们都知道传统搜索都是基于给图片或文字添加 “关键字” 来做搜索的，结果往往非常依赖关键字的弊端，所以才考虑clip

# 改造clip 设置 max_length=8192 支持8K，重新用中文训练

#文字编码+要训练
class TextEncoder(nn.Module):
    def __init__(self, d_out: int) -> None:
        super().__init__()
        self.model = AutoModel.from_pretrained('distilbert-base-multilingual-cased')
        self.tokenizer=AutoTokenizer.from_pretrained('distilbert-base-multilingual-cased')
        self.projection = Projection(Config.transformer_embed_dim, d_out)        

    def forward(self, x):     
        
        text = self.tokenizer( 
            x,
            truncation=True,
            padding=True,
            max_length=8192,   #################################################################设置为 77 表示模型的输入长度限制为 77 个 token  , 设置成：8192 为支持8K输入长度  #################################################################        
            return_tensors="pt",
        ).to(device) 
        x=text["input_ids"]
      
        out = self.model(x)[0] #文本特征: out=torch.Size([128, 25, 768])      
        out = out[:, 0, :]  # get CLS token output  ，获取所有批次 第一行: out=torch.Size([128, 768])
        projected_vec = self.projection(out) #torch.Size([128, 512])
       
        projection_len = torch.norm(projected_vec, dim=-1, keepdim=True)        
        return projected_vec / projection_len
