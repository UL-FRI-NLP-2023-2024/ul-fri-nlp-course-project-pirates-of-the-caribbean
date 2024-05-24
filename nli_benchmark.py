#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import torch
import time
import os
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    TrainerCallback,
    TrainerState,
    TrainerControl
)
from sklearn.model_selection import train_test_split
from datasets import Dataset
from datasets import load_dataset, load_metric
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType, PrefixTuningConfig, IA3Config, PeftModel
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import pandas as pd
import numpy as np
import evaluate
os.environ["WANDB_DISABLED"] = "true"
import sys

seqeval = evaluate.load("seqeval")

TRAIN_PATH = "SI-NLI/train.tsv"
TEST_PATH = "SI-NLI/test.tsv"

class MemoryUsageCallback(TrainerCallback):
    def __init__(self, log_file):
        self.log_file = log_file.replace('/', '_')

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if torch.cuda.is_available():
            allocated_memory = torch.cuda.memory_allocated()
            reserved_memory = torch.cuda.memory_reserved()
            log_message = f"Step {state.global_step} - Allocated Memory: {allocated_memory}, Reserved Memory: {reserved_memory}"
            print(log_message)
            with open('memory_logs/'+self.log_file, "a") as log_f:
                log_f.write(log_message + "\n")

# In[ ]:

seed = int(sys.argv[1])

# In[5]:

train_dataset = pd.read_csv(TRAIN_PATH, sep="\t")
train_dataset, val_dataset = train_test_split(train_dataset, test_size=0.1, random_state=seed)
train_dataset, test_dataset = train_test_split(train_dataset, test_size=0.1, random_state=seed)


def encode_labels(examples):
    label_dict = {"entailment": 0, "neutral": 1, "contradiction": 2}
    # Replace labels in the examples with encoded labels
    print(np.unique(examples["label"]))
    examples["label"] = [label_dict[label] for label in examples["label"]].copy()
    return examples


# In[ ]:


train_dataset = Dataset.from_pandas(encode_labels(train_dataset))
val_dataset = Dataset.from_pandas(encode_labels(val_dataset))
test_dataset = Dataset.from_pandas(encode_labels(test_dataset))


print(train_dataset)
print(val_dataset)
print(test_dataset)


# In[ ]:


# For reference
models = ["EMBEDDIA/sloberta", "bert-base-multilingual-cased"]
model_name = models[0]
num_labels = 3

# In[ ]:


def preprocess_function(examples, tokenizer):
    tokenized = tokenizer(
        examples["premise"],
        examples["hypothesis"],
        padding="max_length",
        truncation=True,
        max_length=492,
    )
    tokenized["label"] = examples["label"]
    return tokenized


# In[ ]:


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),  # Removed .item()
        "f1": f1_score(labels, predictions, average="macro"),
        "precision": precision_score(labels, predictions, average="macro", zero_division=0),
        "recall": recall_score(labels, predictions, average="macro"),
        "f1_per_class": f1_score(labels, predictions, average=None).tolist(),
    }


# In[ ]:

def fine_tune_model(model_name, dataset, model, training_args, finetune_technique="None"):

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    tokenized_train_dataset = train_dataset.map(
        lambda examples: preprocess_function(examples,tokenizer),
        batched=True,
    )
    tokenized_val_dataset = val_dataset.map(
        lambda examples: preprocess_function(examples, tokenizer),
        batched=True,
    )
    tokenized_test_dataset = test_dataset.map(
        lambda examples: preprocess_function(examples, tokenizer),
        batched=True,
    )

    data_collator = DataCollatorWithPadding(
        tokenizer=AutoTokenizer.from_pretrained(model_name),
        padding="max_length",
        max_length=492,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[MemoryUsageCallback(f"NLI_{finetune_technique}_{model_name}")]
    )

    start = time.time()
    trainer.train()
    elapsed_training = time.time() - start

    metrics = trainer.evaluate(tokenized_test_dataset)

    print(f"model: {model_name}, Dataset: SI_NLI, Test Metrics: {metrics}")

    return model, metrics, elapsed_training


def run_bitfit_sloberta(dataset):
    model_name = "EMBEDDIA/sloberta"
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels,
    )

    training_args = TrainingArguments(
        output_dir=f"ner_fully_finetuned_{model_name}",
        learning_rate=2e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=3,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
    )

    # Freeze all the parameters
    for name, param in model.named_parameters():
        param.requires_grad = False

    # Unfreeze the bias parameters
    for name, param in model.named_parameters():
        if 'bias' in name:
            param.requires_grad = True

    _, metrics, elapsed_training = fine_tune_model(
        model_name, dataset, model, training_args, finetune_technique="bitfit"
    )
    print(f"Training time run_lora_sloberta: {elapsed_training}")
    current_time = time.strftime("%Y-%m-%d-%H-%M-%S")
    with open("results_run_bitfit_sloberta.csv", "a") as f:
        f.write(
            f"{current_time},{model_name},NLI, {metrics},{elapsed_training}\n"
        )


run_bitfit_sloberta(dataset = {"train": train_dataset, "val": val_dataset, "test": test_dataset})


def run_bitfit_bert(dataset):
    model_name = "bert-base-multilingual-cased"
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels,
    )

    training_args = TrainingArguments(
        output_dir=f"ner_fully_finetuned_{model_name}",
        learning_rate=2e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=3,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
    )

    # Freeze all the parameters
    for name, param in model.named_parameters():
        param.requires_grad = False

    # Unfreeze the bias parameters
    for name, param in model.named_parameters():
        if 'bias' in name:
            param.requires_grad = True

    _, metrics, elapsed_training = fine_tune_model(
        model_name, dataset, model, training_args, finetune_technique="bitfit"
    )
    print(f"Training time run_lora_sloberta: {elapsed_training}")
    current_time = time.strftime("%Y-%m-%d-%H-%M-%S")
    with open("results_run_bitfit_bert.csv", "a") as f:
        f.write(
            f"{current_time},{model_name},NLI, {metrics},{elapsed_training}\n"
        )

run_bitfit_bert(dataset = {"train": train_dataset, "val": val_dataset, "test": test_dataset})


def fully_finetune_sloberta():
    model_name = "EMBEDDIA/sloberta"
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels,
    )

    training_args = TrainingArguments(
        output_dir=f"ner_fully_finetuned_{model_name}",
        learning_rate=2e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=3,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="no",
        load_best_model_at_end=True,
    )


    dataset = {"train": train_dataset, "val": val_dataset, "test": test_dataset}
    model, metrics, elapsed_training =  fine_tune_model(model_name, dataset, model, training_args=training_args, finetune_technique="Fully-FT")
    current_time = time.strftime("%Y-%m-%d-%H-%M-%S")
    with open("results_fully_finetune_sloberta.csv", "a") as f:
        f.write(
            f"{current_time},{model_name},NLI, {metrics},{elapsed_training}\n"
        )
    print(f"Training time fully_finetune_sloberta: {elapsed_training}")

fully_finetune_sloberta()


# In[ ]:


def fully_finetune_bert():
    model_name = "bert-base-multilingual-cased"
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels,
    )

    training_args = TrainingArguments(
        output_dir=f"ner_fully_finetuned_{model_name}",
        learning_rate=2e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=3,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="no",
        load_best_model_at_end=True,
    )

    dataset = {"train": train_dataset, "val": val_dataset, "test": test_dataset}
    model, metrics, elapsed_training =  fine_tune_model(model_name, dataset, model, training_args=training_args, finetune_technique="Full-FT")
    current_time = time.strftime("%Y-%m-%d-%H-%M-%S")
    with open("results_fully_finetune_bert.csv", "a") as f:
        f.write(
            f"{current_time},{model_name},NLI, {metrics},{elapsed_training}\n"
        )
    print(f"Training time fully_finetune_bert: {elapsed_training}")

fully_finetune_bert()


# In[ ]:


def run_lora_sloberta(dataset):
    model_name = "EMBEDDIA/sloberta"
    task_type = TaskType.SEQ_CLS
    training_args = TrainingArguments(
        output_dir=f"ner_lora_finetuned_{model_name}",
        learning_rate=1e-3,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=3,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="no",
        load_best_model_at_end=True,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels,
    )

    target_modules = (
        [
            "roberta.encoder.layer." + str(i) + ".attention.self.query"
            for i in range(model.config.num_hidden_layers)
        ]
        + [
            "roberta.encoder.layer." + str(i) + ".attention.self.key"
            for i in range(model.config.num_hidden_layers)
        ]
        + [
            "roberta.encoder.layer." + str(i) + ".attention.self.value"
            for i in range(model.config.num_hidden_layers)
        ]
        + [
            "roberta.encoder.layer." + str(i) + ".attention.output.dense"
            for i in range(model.config.num_hidden_layers)
        ]
    )
    
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS, inference_mode=False, r=16, lora_alpha=16, lora_dropout=0.1, bias="all"
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        task_type=task_type,
        bias="none",
        target_modules=target_modules,
    )

    model = get_peft_model(model, lora_config)

    _, metrics, elapsed_training = fine_tune_model(
        model_name, dataset, model, training_args, finetune_technique="LORA"
    )
    print(f"Training time run_lora_sloberta: {elapsed_training}")
    current_time = time.strftime("%Y-%m-%d-%H-%M-%S")
    with open("results_run_lora_sloberta.csv", "a") as f:
        f.write(
            f"{current_time},{model_name},NLI, {metrics},{elapsed_training}\n"
        )
    print(f"Training time run_lora_sloberta: {elapsed_training}")


# In[ ]:


run_lora_sloberta(dataset = {"train": train_dataset, "val": val_dataset, "test": test_dataset})


# In[ ]:


def run_lora_bert(dataset):
    model_name = "bert-base-multilingual-cased"
    task_type = TaskType.SEQ_CLS
    training_args = TrainingArguments(
        output_dir=f"ner_lora_finetuned_{model_name}",
        learning_rate=1e-3,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=3,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="no",
        load_best_model_at_end=True,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels
    )

    target_modules = (
        [
            "bert.encoder.layer." + str(i) + ".attention.self.query"
            for i in range(model.config.num_hidden_layers)
        ]
        + [
            "bert.encoder.layer." + str(i) + ".attention.self.key"
            for i in range(model.config.num_hidden_layers)
        ]
        + [
            "bert.encoder.layer." + str(i) + ".attention.self.value"
            for i in range(model.config.num_hidden_layers)
        ]
        + [
            "bert.encoder.layer." + str(i) + ".attention.output.dense"
            for i in range(model.config.num_hidden_layers)
        ]
    )
    
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS, inference_mode=False, r=16, lora_alpha=16, lora_dropout=0.1, bias="all"
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        task_type=task_type,
        bias="none",
        target_modules=target_modules,
    )

    model = get_peft_model(model, lora_config)

    _, metrics, elapsed_training = fine_tune_model(
        model_name, dataset, model, training_args
    )
    print(f"Training time: {elapsed_training}")
    current_time = time.strftime("%Y-%m-%d-%H-%M-%S")
    with open("results_run_lora_bert.csv", "a") as f:
        f.write(
            f"{current_time},{model_name},Dependency LB, {metrics},{elapsed_training}\n"
        )
    print(f"Training time: {elapsed_training}")


# In[ ]:


run_lora_bert(dataset = {"train": train_dataset, "val": val_dataset, "test": test_dataset})


# In[ ]:


def run_prefix_tune_sloberta(dataset):
    model_name = "EMBEDDIA/sloberta"

    task_type = TaskType.SEQ_CLS
    training_args = TrainingArguments(
        output_dir=f"ner_prefix_tunning_finetuned_{model_name}",
        learning_rate=1e-2,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=3,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="no",
        load_best_model_at_end=True,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels
    )

    target_modules = (
        [
            "roberta.encoder.layer." + str(i) + ".attention.self.query"
            for i in range(model.config.num_hidden_layers)
        ]
        + [
            "roberta.encoder.layer." + str(i) + ".attention.self.key"
            for i in range(model.config.num_hidden_layers)
        ]
        + [
            "roberta.encoder.layer." + str(i) + ".attention.self.value"
            for i in range(model.config.num_hidden_layers)
        ]
        + [
            "roberta.encoder.layer." + str(i) + ".attention.output.dense"
            for i in range(model.config.num_hidden_layers)
        ]
    )

    prefix_config = PrefixTuningConfig(task_type="SEQ_CLS", num_virtual_tokens=20)

    model = get_peft_model(model, prefix_config)

    _, metrics, elapsed_training = fine_tune_model(
        model_name, dataset, model, training_args, finetune_technique="P-tunning"
    )
    print(f"Training time run_prefix_tune_sloberta: {elapsed_training}")
    current_time = time.strftime("%Y-%m-%d-%H-%M-%S")
    with open("results_run_prefix_tune_sloberta.csv", "a") as f:
        f.write(
            f"{current_time},{model_name},NLI, {metrics},{elapsed_training}\n"
        )


# In[ ]:


run_prefix_tune_sloberta(dataset = {"train": train_dataset, "val": val_dataset, "test": test_dataset})


# In[ ]:


def run_prefix_tune_bert(dataset):
    model_name = "bert-base-multilingual-cased"

    task_type = TaskType.SEQ_CLS
    training_args = TrainingArguments(
        output_dir=f"ner_prefix_tunning_finetuned_{model_name}",
        learning_rate=1e-2,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=3,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="no",
        load_best_model_at_end=True,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels
    )

    target_modules = (
        [
            "bert.encoder.layer." + str(i) + ".attention.self.query"
            for i in range(model.config.num_hidden_layers)
        ]
        + [
            "bert.encoder.layer." + str(i) + ".attention.self.key"
            for i in range(model.config.num_hidden_layers)
        ]
        + [
            "bert.encoder.layer." + str(i) + ".attention.self.value"
            for i in range(model.config.num_hidden_layers)
        ]
        + [
            "bert.encoder.layer." + str(i) + ".attention.output.dense"
            for i in range(model.config.num_hidden_layers)
        ]
    )

    prefix_config = PrefixTuningConfig(task_type="SEQ_CLS", num_virtual_tokens=20)

    model = get_peft_model(model, prefix_config)

    _, metrics, elapsed_training = fine_tune_model(
        model_name, dataset, model, training_args, finetune_technique="P-tunning"
    )
    print(f"Training time run_prefix_tune_bert: {elapsed_training}")
    current_time = time.strftime("%Y-%m-%d-%H-%M-%S")
    with open("results_run_prefix_tune_bert.csv", "a") as f:
        f.write(
            f"{current_time},{model_name},NLI, {metrics},{elapsed_training}\n"
        )


# In[ ]:


run_prefix_tune_bert(dataset = {"train": train_dataset, "val": val_dataset, "test": test_dataset})


# In[ ]:


def run_ia3_sloberta(dataset):
    model_name = "EMBEDDIA/sloberta"
    task_type = TaskType.SEQ_CLS
    training_args = TrainingArguments(
        output_dir=f"ner-ia3-{model_name}",
        learning_rate=1e-2,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=3,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="no",
        load_best_model_at_end=True,
    )
    model = AutoModelForSequenceClassification.from_pretrained( 
        model_name, num_labels=num_labels
    )
    model = prepare_model_for_kbit_training(model, task_type)

    feed_forward_modules = [
        "roberta.encoder.layer." + str(i) + ".intermediate.dense"
        for i in range(model.config.num_hidden_layers)
    ]

    target_modules = (
        [
            "roberta.encoder.layer." + str(i) + ".attention.self.query"
            for i in range(model.config.num_hidden_layers)
        ]
        + [
            "roberta.encoder.layer." + str(i) + ".attention.self.key"
            for i in range(model.config.num_hidden_layers)
        ]
        + [
            "roberta.encoder.layer." + str(i) + ".attention.self.value"
            for i in range(model.config.num_hidden_layers)
        ]
        + [
            "roberta.encoder.layer." + str(i) + ".attention.output.dense"
            for i in range(model.config.num_hidden_layers)
        ]
        + feed_forward_modules
    )

    ia3_config = IA3Config(task_type=task_type, feedforward_modules = feed_forward_modules, target_modules=target_modules)

    model = get_peft_model(model, ia3_config)

    _, metrics, elapsed_training = fine_tune_model(
        model_name, dataset, model, training_args, finetune_technique="IA2"
    )
    print(f"Training time run_ia3_sloberta: {elapsed_training}")
    current_time = time.strftime("%Y-%m-%d-%H-%M-%S")
    with open("results_run_ia3_sloberta.csv", "a") as f:
        f.write(
            f"{current_time},{model_name},NLI, {metrics},{elapsed_training}\n"
        )


# In[ ]:


run_ia3_sloberta(dataset = {"train": train_dataset, "val": val_dataset, "test": test_dataset})


# In[ ]:


def run_ia3_bert(dataset):
    model_name = "bert-base-multilingual-cased"
    task_type = TaskType.SEQ_CLS
    training_args = TrainingArguments(
        output_dir=f"ner-ia3-{model_name}",
        learning_rate=1e-1,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=3,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="no",
        load_best_model_at_end=True,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels
    )
    model = prepare_model_for_kbit_training(model, task_type)

    feed_forward_modules = [
        "roberta.encoder.layer." + str(i) + ".intermediate.dense"
        for i in range(model.config.num_hidden_layers)
    ]

    target_modules = (
        [
            "bert.encoder.layer." + str(i) + ".attention.self.query"
            for i in range(model.config.num_hidden_layers)
        ]
        + [
            "bert.encoder.layer." + str(i) + ".attention.self.key"
            for i in range(model.config.num_hidden_layers)
        ]
        + [
            "bert.encoder.layer." + str(i) + ".attention.self.value"
            for i in range(model.config.num_hidden_layers)
        ]
        + [
            "bert.encoder.layer." + str(i) + ".attention.output.dense"
            for i in range(model.config.num_hidden_layers)
        ]
        + feed_forward_modules
    )

    ia3_config = IA3Config(task_type=task_type, feedforward_modules = feed_forward_modules, target_modules=target_modules)

    model = get_peft_model(model, ia3_config)

    _, metrics, elapsed_training = fine_tune_model(
        model_name, dataset, model, training_args, finetune_technique="IA3"
    )
    print(f"Training time run_ia3_bert: {elapsed_training}")
    current_time = time.strftime("%Y-%m-%d-%H-%M-%S")
    with open("results_run_ia3_bert.csv", "a") as f:
        f.write(
            f"{current_time},{model_name},NLI, {metrics},{elapsed_training}\n"
        )


# In[ ]:


run_ia3_bert(dataset = {"train": train_dataset, "val": val_dataset, "test": test_dataset})
