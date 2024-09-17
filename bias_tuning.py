from datasets import load_dataset

import numpy as np 
import evaluate

import torch 
from transformers import Trainer
from transformers import TrainingArguments

import low_rank
from transformers import ViTImageProcessor
from transformers import ViTForImageClassification


processor = ViTImageProcessor.from_pretrained('./vit-base-patch16-224')

def process_example(example):
    if example['image'].mode != 'RGB':
        example['image'] = example['image'].convert('RGB')
    inputs = processor(example['image'], return_tensors='pt')
    inputs['labels'] = example['label']
    return inputs

trainset = load_dataset('/data/jc/dataset/imagenet-1k', split='train', streaming=True)
valset = load_dataset('/data/jc/dataset/imagenet-1k', split='validation', streaming=True)

prepared_trainset = trainset.map(process_example)
prepared_valset = valset.map(process_example)




model = ViTForImageClassification.from_pretrained('./vit-base-patch16-224')

count_params = sum(p.numel() for p in model.parameters())
model_lr = low_rank.ModuleLowRank(compress_ratio=2, 
                                name_omit=['norm', 'head', 'patch_embed', 'downsample'],
                                is_approximate=True)
model = model_lr(model)
count_lr_params = sum(p.numel() for p in model.parameters())

print(f'Original model params: {count_params}, Low rank model params: {count_lr_params}')

# only set bias to be optimized
for name, param in model.named_parameters():
    if 'bias' in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

count_learn_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Learnable params: {count_learn_params}')

metric = evaluate.load('accuracy')

def compulate_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return metric.compute(predictions=preds, references=labels)


training_args = TrainingArguments(    
    # train epochs and batch size
    num_train_epochs=50,              # total number of training epochs
    per_device_train_batch_size=256,  # batch size per device during training
    per_device_eval_batch_size=256,   # batch size for evaluation
    max_steps= int(50*1.2e7/128),

    # learning rate and warmup steps
    learning_rate=5e-4,               # initial learning rate
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    
    # logging
    output_dir='./results',          # output directory
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
    report_to='tensorboard',

    # evaluation
    eval_steps=1000,
    save_steps=1000,
    evaluation_strategy='steps',
    save_strategy='steps',
    save_total_limit=3,
    load_best_model_at_end=True,
)


def collect_fn(batch):
    batch = {
        'pixel_values': torch.cat([x['pixel_values'] for x in batch], dim=0),
        'labels': torch.tensor([x['labels'] for x in batch])
    }
    return batch

trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=prepared_trainset,         # training dataset
    eval_dataset=prepared_valset,             # evaluation dataset
    tokenizer=processor,
    data_collator=collect_fn,
    compute_metrics=compulate_metrics,
)

trainer.train()
trainer.evaluate()


