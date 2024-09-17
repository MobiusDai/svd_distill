import os 
from PIL import Image
from tqdm import tqdm 

import torch 
from torch.utils.data import DataLoader
from dataset.classes import IMAGENET2012_CLASSES
from custom_dataset import ImageNet1K
from transformers import ViTImageProcessor, ViTForImageClassification

from transformers.models.vit.modeling_vit import ViTSelfAttention
from sparse_attention import SparseSelfAttention

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default="./vit-base-patch16-224")
parser.add_argument("--compress_ratio", type=float, default=8)
parser.add_argument("--sparse_ratio", type=float, default=0.1)
args = parser.parse_args()

processor = ViTImageProcessor.from_pretrained(args.model_path)
model = ViTForImageClassification.from_pretrained(args.model_path)

total_params = sum(p.numel() for p in model.parameters())

'''sparse attention'''
for name, module in model.named_modules():
    if isinstance(module, ViTSelfAttention):
        print(name)
        module.__class__ = SparseSelfAttention
        module.ratio = args.sparse_ratio



'''low rank approximation'''

import low_rank

model_lr = low_rank.ModuleLowRank(compress_ratio=args.compress_ratio, 
                                name_omit=['norm', 'head', 'patch_embed', 'downsample'],
                                name_include= ['query', 'key'], 
                                is_approximate=True)
model = model_lr(model)
low_rank_params = sum(p.numel() for p in model.parameters())

print(f'Original model params: {total_params}, low rank model params: {low_rank_params}, compression ratio: {low_rank_params/total_params}')


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

val_dataset = ImageNet1K(image_path='./dataset/val_data/', labels=IMAGENET2012_CLASSES, transform=processor)
val_loader = DataLoader(
    dataset= val_dataset,
    batch_size= 120,
    num_workers= 8
)


# test the accuracy of vision transformer
accurate = 0
count = 0 

for i, batch in enumerate(val_loader):
    image = batch[0]['pixel_values'].squeeze(1).to('cuda')
    label = batch[1].to('cuda')
    pred = model(image).logits.argmax(dim=1)
    
    accurate += (pred == label).sum()
    count += image.shape[0]
    
    if i % 50 == 0:
        print(f'step {i}/ {len(val_loader)}, accuracy: {accurate/count}')


print(f"Accuracy: {accurate/len(val_dataset)}")

with open(f'experiments/result-{args.compress_ratio}-{args.sparse_ratio}.txt', 'w') as f:
    f.write(f"Accuracy: {accurate/len(val_dataset)}")
