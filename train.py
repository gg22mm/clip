'''
pip install clip -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install openai-clip -i https://pypi.tuna.tsinghua.edu.cn/simple
'''
import torch
import os
import torch
import subprocess
from torch.utils.data import DataLoader
from dataclasses import dataclass

# 模型
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from transformers import AutoModel, AutoTokenizer, BertTokenizer

# 基本配置
device = "cuda" if torch.cuda.is_available() else "cpu"

@dataclass
class Config:   
    embed_dim: int = 512                # 模型向量输出 Embedding dimension - 
    transformer_embed_dim: int = 768    # 模型 Transformer embedding dimension -         
    batch_size: int = 128               # Batch size


# ########################## 读取数据集 ##########################

# from src.clip_dl import CocoDataset, Flickr30kDataset

import json
from PIL import Image
from torch.utils.data import Dataset
import collections
from torchvision import transforms
from datasets import load_dataset
import clip
from transformers import CLIPProcessor, CLIPModel

class CocoDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]
        )
        annotations_dir = os.path.join(root_dir, "annotations")
        annotation_file = os.path.join(
            annotations_dir, "annotations/captions_val2017.json" #captions_train2017.json  //annotations_dir, "annotations", "captions_val2017.json" 
        )

        self.caption_list, self.image_path_list = self.load_annotations(annotation_file)

    def load_annotations(self, annotation_file):
        with open(annotation_file, "r") as f:
            annotations = json.load(f)["annotations"]

        image_path_to_caption = collections.defaultdict(list)
        for element in annotations:
            caption = f"{element['caption'].lower().rstrip('.')}"
            image_path = os.path.join(
                self.root_dir,
                # "val2017", #train2017
                "val2017/val2017", #train2017
                "%012d.jpg" % (element["image_id"]),
            )
            image_path_to_caption[image_path].append(caption)
        image_paths = list(image_path_to_caption.keys())
        caption_list, image_path_list = self.training_list(
            image_paths, image_path_to_caption
        )

        return caption_list, image_path_list

    def training_list(self, image_paths, image_path_to_caption):
        captions_per_image = 2
        caption_list = []
        image_path_list = []
        for image_path in image_paths:
            captions = image_path_to_caption[image_path][:captions_per_image]
            caption_list.extend(captions)
            image_path_list.extend([image_path] * len(captions))

        return caption_list, image_path_list

    def __len__(self):
        return len(self.caption_list)

    def __getitem__(self, idx):
        image_path = self.image_path_list[idx]
        caption = self.caption_list[idx]

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return {"image": image, "caption": caption}


coco_dataset = True

# Create the CLIP dataset
if coco_dataset:
    
    # if not "datasets" in os.listdir():
    #     print("coco dataset is not downloaded! running the downloading script ....")
    #     subprocess.run(["python", "src/download_coco_data.py"])

    clip_dataset = CocoDataset(root_dir="/kaggle/input/mscoco-2017-trainval-annotations/")
else:
    clip_dataset = Flickr30kDataset()


# print(clip_dataset[0])
# exit()


# Create the DataLoader
clip_dataloader = DataLoader(  clip_dataset, batch_size=Config.batch_size, shuffle=True, num_workers=0)

# print( next(iter(clip_dataloader)) )
# exit() 

# ########################## 损失函数 ##########################

def CLIP_loss(logits):
    """
    Calculate a custom cross-entropy loss.

    Args:
    - logits (torch.Tensor): The input tensor containing unnormalized logits.

    Returns:
    - torch.Tensor: The computed custom cross-entropy loss.

    Example:
    >>> logits = torch.rand((batch_size, num_classes))
    >>> loss = CLIP_loss(logits)
    """

    n = logits.shape[1] #logits = torch.Size([128, 128]) , 那么 n=128  

    '''
    torch.arange()，torch.range()
    按序输出整数; torch.arange()默认从0开始，一直到输入的数-1，向量内部元素默认为整数；torch.range()需输入起始值和最终值，生成的向量包括输入值；向量内部元素默认为浮点数；
    t = torch.arange(7)    # t = tensor([0, 1, 2, 3, 4, 5, 6])
    t = torch.range(0, 7)  # t = tensor([0., 1., 2., 3., 4., 5., 6., 7.])
    '''
    # y - 默认从0开始，一直到输入的数-1
    labels = torch.arange(n)  #torch.Size([128])
       
    # 将logits带到cpu - bring logits to cpu
    logits = logits.to("cpu") #torch.Size([128, 128])

    # 计算沿轴0和1的交叉熵损失  Calculate cross entropy losses along axis 0 and 1
    loss_i = F.cross_entropy(logits.transpose(0, 1), labels, reduction="mean") # transpose 只能对两个维度进行转换
    loss_t = F.cross_entropy(logits, labels, reduction="mean")

    # Calculate the final loss
    loss = (loss_i + loss_t) / 2

    return loss

# 韵律学
def metrics(similarity):
    y = torch.arange(len(similarity)).to(similarity.device)
    img2cap_match_idx = similarity.argmax(dim=1)
    cap2img_match_idx = similarity.argmax(dim=0)

    img_acc = (img2cap_match_idx == y).float().mean()
    cap_acc = (cap2img_match_idx == y).float().mean()

    return img_acc, cap_acc




##############################


torch.cuda.empty_cache() # 清除缓存

# ########################## 一、模型定义 ##########################

# 只训练这个+文字编码 - 多模态嵌入空间,自己定义的clip模型,优化的是这些参数 
class Projection(nn.Module):
    def __init__(self, d_in: int, d_out: int, p: float = 0.5) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_in, d_out, bias=False)  #768,512
        self.linear2 = nn.Linear(d_out, d_out, bias=False) #512,d_out
        self.layer_norm = nn.LayerNorm(d_out)
        self.drop = nn.Dropout(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embed1 = self.linear1(x)
        embed2 = self.drop(self.linear2(F.gelu(embed1)))
        embeds = self.layer_norm(embed1 + embed2)

        # print(embed1.shape,embed2.shape,embeds.shape) #torch.Size([128, 512]) torch.Size([128, 512]) torch.Size([128, 512])       
        return embeds

#图片编码+不训练
class ImageEncoder(nn.Module):
    def __init__(self, d_out: int) -> None:
        super().__init__()
        
        # 方式一
        # base = models.resnet34(pretrained=True)  #原： (fc): Linear(in_features=512, out_features=1000, bias=True)             
        # d_in = base.fc.in_features #获得原线性全连接输入:d_in=512 ,  其实就是达到：base.fc = nn.Linear(base.fc.in_features, d_out=512) #替换成自己的Linear层        
        # base.fc = nn.Identity() #（自动 这个把输出的类别 设置等于 输入的类别维度） - 手动其实就是：base.fc = nn.Linear(base.fc.in_features=512, d_out=512) #替换成自己的Linear层，这里的512=传进来的d_out        
        # self.model = base 
        # self.projection = Projection(d_in, d_out) #只训练这个 - 多模态嵌入空间,自己定义的clip模型,优化的是这些参数 ：  d_in=768 , d_out=512   
        # # 冻结 resnet 模型不做为训练对象
        # for p in self.model.parameters():
        #     p.requires_grad = False

        # 方式二，直接使用 openai 的 clip 并且冻结，这样以后可以支持sd生成图片
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval()   
        d_in = self.model.text_projection.in_features     
        self.projection = Projection(d_in, d_out) #只训练这个 - 多模态嵌入空间,自己定义的clip模型,优化的是这些参数 ：  d_in=768 , d_out=512

        # 冻结 resnet 模型不做为训练对象
        for p in self.model.parameters():
            p.requires_grad = False

    def forward(self, x):   
        
        # x.shape = torch.Size([128, 3, 224, 224])
        #print('-------------2-----------------',x.device)
        # 方式一
        # projected_vec = self.projection(self.model(x)) #self.model(x)=torch.Size([128, 512]) , projected_vec.shape = torch.Size([128, 512])    

        # 方式二，直接使用 openai 的 clip 并且冻结，这样以后可以支持sd生成图片
        image = self.processor(images=x, return_tensors="pt").to(device) #图片格式化标准输入 # image = image.convert('RGB')  
        #print('-------------3-----------------')
        
        image_features = self.model.get_image_features(**image)   #图片特征 - 这人可以存到es中 叫：Knn相似度查询 # 加载CLIP的image encoder model
        # print('-----------image_features-=========:',image_features.shape,model) #torch.Size([128, 512])        
        projected_vec = self.projection(image_features) #self.model(x)=torch.Size([128, 512]) , projected_vec.shape = torch.Size([128, 512]) 
        
        # 继续
        projection_len = torch.norm(projected_vec, dim=-1, keepdim=True) #torch.norm()是对输入的tensor求对应的范数 ，  输出： torch.Size([128, 1])        
        return projected_vec / projection_len #torch.Size([128, 512])

#文字编码+要训练
class TextEncoder(nn.Module):
    def __init__(self, d_out: int) -> None:
        super().__init__()
        self.model = AutoModel.from_pretrained('distilbert/distilbert-base-multilingual-cased')
        self.tokenizer=AutoTokenizer.from_pretrained('distilbert/distilbert-base-multilingual-cased')
        self.projection = Projection(Config.transformer_embed_dim, d_out)
        
        # # 冻结 distilbert-base-multilingual-cased 模型不做为训练对象
        # for p in self.model.parameters():
        #     p.requires_grad = False

    def forward(self, x):         
        # 直接文字转成token id , 指定长度来获取，要不太长了会爆
        text = self.tokenizer( 
            x,########## 重要的东西在这里了，获取这个字段进行转成id的 
            truncation=True,
            padding=True,
            max_length=8192,   #设置为 77 表示模型的输入长度限制为 77 个 token  , 设置成：8192 为支持8K输入长度         
            return_tensors="pt",
        ).to(device) 
        x=text["input_ids"]
        # print('------输入文字x转成token id:',x.shape,x) # torch.Size([1, 15])
        # exit()         

        out = self.model(x)[0] #文本特征: out=torch.Size([128, 25, 768])      
        out = out[:, 0, :]  # get CLS token output  ，获取所有批次 第一行: out=torch.Size([128, 768])
        projected_vec = self.projection(out) #torch.Size([128, 512])
       
        projection_len = torch.norm(projected_vec, dim=-1, keepdim=True)        
        return projected_vec / projection_len


class ClipModel(nn.Module):
    def __init__(self, lr: float = 1e-3) -> None:
        super().__init__()
        self.image_Encoder = ImageEncoder(Config.embed_dim) #图片编码
        self.text_Encoder = TextEncoder(Config.embed_dim)  #文字编码       
        self.lr = lr        

    def forward(self, images, text):
        #print('-----------1-------------------',image.device)
        
        #图片编码
        image_embed = self.image_Encoder(images)        

        #文字编码
        text_embed = self.text_Encoder(text)
        
        # 计算余弦相似度
        similarity = text_embed @ image_embed.T  #@矩阵乘法：  文字编码 乘以 图片编码的转置 #torch.Size([128, 128])        
        loss = CLIP_loss(similarity)                #求loss的传参 = 文字编码 与 图片编码的转置 相矩阵，为什么这样，是不是这样后就可以在到同时传两个参数过去，然后他们又可以相产生一些关系，这可能是作者的 做实验得到的一些成果
        img_acc, cap_acc = metrics(similarity)      #韵律学
        return loss, img_acc, cap_acc

    # 只获取图像编码
    def encode_image(self,images):
        return self.image_Encoder(images)

    # 只获取文本编码
    def encode_text(self,text):
        return self.text_Encoder(text)


# 创建clip模型
model = ClipModel().to(device)
# print(model)
# # exit()

# ########################## 开始训练 ##########################


# Define optimizer
optimizer = torch.optim.Adam(
    [
        {"params": model.image_Encoder.parameters()},
        {"params": model.text_Encoder.parameters()},
    ],
    lr=model.lr,
)

# Dummy training and validation loops
num_epochs = 50
batch_zero = True
for epoch in range(num_epochs):
    model.train()
    for batch in clip_dataloader:
        image = batch["image"].to(device)
        text = batch["caption"]
        # images, text = batch
        loss, img_acc, cap_acc = model(image, text)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_zero:
            print(f"Epoch [{0}/{num_epochs}], Batch Loss: {loss.item()}")
            batch_zero = False

    # Print training statistics
    print(f"Epoch [{epoch+1}/{num_epochs}], Batch Loss: {loss.item()}")
    
    # 模型保存
    torch.save(model.state_dict(), './output_'+str(epoch)+'.pth')

print("Training complete.")
