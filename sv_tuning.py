from datasets import load_dataset

import numpy as np 

import torch 
from transformers import Trainer
from transformers import TrainingArguments

import low_rank
from transformers import ViTImageProcessor
from transformers import ViTForImageClassification


train_dataset = load_dataset("/data/jc/datasets/cifar-10", split="train", streaming=True)
val_dataset = load_dataset("/data/jc/datasets/cifar-10", split="test", streaming=True)

processor = ViTImageProcessor.from_pretrained('./weights/vit-base-patch16-224-in21k-finetuned-cifar10/')

def preprocess_function(item):
    # Resize the input image to the model's size
    inputs = processor(images=item["img"], return_tensors="pt")
    inputs["labels"] = item["label"]
    return inputs 

train_dataset = train_dataset.map(preprocess_function, remove_columns=["img"], batched=True)
val_dataset = val_dataset.map(preprocess_function, remove_columns=["img"], batched=True)


def collect_fn(batch):
    batch = {    
        'pixel_values': torch.stack([x['pixel_values'] for x in batch], dim=0),
        'labels': torch.tensor([x['labels'] for x in batch])    
    }
    return batch


def compulate_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    accuracy = np.mean(preds == labels)
    return {"accuracy": accuracy}

model = ViTForImageClassification.from_pretrained('./weights/vit-base-patch16-224-in21k-finetuned-cifar10')

count_params = sum(p.numel() for p in model.parameters())
model_lr_transform = low_rank.ModuleLowRank(compress_ratio=2, 
                                name_omit=['norm', 'head', 'patch_embed', 'downsample'],
                                is_approximate=True)
low_rank_model = model_lr_transform(model)
count_lr_params = sum(p.numel() for p in low_rank_model.parameters())

print(f'Original model params: {count_params}, Low rank model params: {count_lr_params}')

train_params = 0
for name, param in low_rank_model.named_parameters():
    param.requires_grad = False
    if any(nd in name for nd in ['norm', 'head', 'patch_embed', 'downsample']):
        continue
    if 'sv' in name:
        param.requires_grad = True
        train_params += param.numel()

print(f'Trainable params: {train_params}')

batch_size = 200

trainer = Trainer(
    model=low_rank_model,
    data_collator=collect_fn,
    train_dataset = train_dataset,
    eval_dataset = val_dataset,
    compute_metrics=compulate_metrics,
    args = TrainingArguments(
        # train epochs and batch size
        num_train_epochs=50,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        max_steps = int(50*50000/(batch_size*4)),

        # learning rate and warmup steps
        learning_rate=1e-3,
        warmup_steps=100,
        weight_decay=0.01,

        # logging
        output_dir="./results",
        logging_dir="./logs",
        logging_steps=10,
        report_to='tensorboard',

        # evaluation
        eval_steps=50,
        save_steps=50,
        evaluation_strategy='steps',
        save_strategy='steps',
        save_total_limit=3,
        load_best_model_at_end=True,
    )
)

trainer.train()
trainer.evaluate()