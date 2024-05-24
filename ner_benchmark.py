#!/usr/bin/env python
# coding: utf-8

# In[3]:

import torch
import time
import os
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
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
import sys
seqeval = evaluate.load("seqeval")
os.environ["WANDB_DISABLED"] = "true"


class MemoryUsageCallback(TrainerCallback):
    def __init__(self, log_file):
        self.log_file = log_file.replace('/', '_')

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if torch.cuda.is_available():
            allocated_memory = torch.cuda.memory_allocated()
            reserved_memory = torch.cuda.memory_reserved()
            log_message = f"Step {state.global_step} - Allocated Memory: {allocated_memory}, Reserved Memory: {reserved_memory}"
            print(log_message)
            with open("memory_logs/"+self.log_file, "a") as log_f:
                log_f.write(log_message + "\n")

# In[4]:


dataset = load_dataset("cjvt/ssj500k", "named_entity_recognition")


seed = int(sys.argv[1])

# In[5]:


train_dataset = dataset["train"].to_pandas()
train_dataset, test_dataset = train_test_split(
    train_dataset, test_size=0.2, random_state=seed
)
train_dataset, val_dataset = train_test_split(
    train_dataset, test_size=0.1, random_state=seed
)


# In[6]:


id2label = {0: 'O',
 1: 'B-LOC',
 2: 'I-LOC',
 3: 'B-ORG',
 4: 'I-ORG',
 5: 'B-PER',
 6: 'I-PER',
 7: 'B-MISC',
 8: 'I-MISC'
}

label2id = {label: id for id,label in id2label.items()}


# In[7]:


train_dataset = Dataset.from_pandas(train_dataset)
val_dataset = Dataset.from_pandas(val_dataset)
test_dataset = Dataset.from_pandas(test_dataset)


print(train_dataset)
print(val_dataset)
print(test_dataset)


# In[8]:


# For reference
models = ["EMBEDDIA/sloberta", "bert-base-multilingual-cased"]
model_name = models[0]


# In[9]:


tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
from transformers import DataCollatorForTokenClassification
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)


# In[10]:


def tokenize_and_align_labels(examples, tokenizer):
    tokenized_inputs = tokenizer(examples["words"], truncation=True, is_split_into_words=True)
    
    labels = []
    for i, label in enumerate(examples[f"ne_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:  # Set the special tokens to -100.
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                label_ids.append(label2id[label[word_idx]])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


# In[11]:


def preprocess_function(examples, tokenizer):
    tokenized_inputs = tokenize_and_align_labels(examples, tokenizer)
    return tokenized_inputs


# In[12]:


tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
tokenized_train = train_dataset.map(lambda example: preprocess_function(example, tokenizer), batched=True)


# In[13]:


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [id2label[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    result_metrics = {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }
    return result_metrics


# In[14]:


def fine_tune_model(model_name, dataset, model, training_args, finetune_technique="None"):
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    print(dataset)
    
    tokenized_train_dataset = dataset["train"].map(
        lambda examples: preprocess_function(examples, tokenizer),
        batched=True,
    )
    tokenized_val_dataset = dataset["val"].map(
        lambda examples: preprocess_function(examples, tokenizer),
        batched=True,
    )
    tokenized_test_dataset = dataset["test"].map(
        lambda examples: preprocess_function(examples, tokenizer),
        batched=True,
    )

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[MemoryUsageCallback(f"SSJ500K_{finetune_technique}_{model_name}")]
    )

    start = time.time()
    trainer.train()
    elapsed_training = time.time() - start

    metrics = trainer.evaluate(tokenized_test_dataset)

    print(f"model: {model_name}, Dataset: ssj500k, Test Metrics: {metrics}")

    return model, metrics, elapsed_training



# In[16]:


def fully_finetune_sloberta():
    model_name = "EMBEDDIA/sloberta"
    model = AutoModelForTokenClassification.from_pretrained(
        model_name, num_labels=9, id2label=id2label, label2id=label2id
    )

    training_args = TrainingArguments(
        output_dir=f"ner_checkpoints/ner_fully_finetuned_{model_name}",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit = 2,
        load_best_model_at_end=True,
    )


    dataset = {"train": train_dataset, "val": val_dataset, "test": test_dataset}
    model, metrics, elapsed_training =  fine_tune_model(model_name, dataset, model, training_args=training_args, finetune_technique="Full-FT")
    current_time = time.strftime("%Y-%m-%d-%H-%M-%S")
    with open("results_fully_finetune_sloberta.csv", "a") as f:
        f.write(
            f"{current_time},{model_name},SSJ500-NER, {metrics},{elapsed_training}\n"
        )
    print(f"Training time fully_finetune_sloberta: {elapsed_training}")

fully_finetune_sloberta()


# In[ ]:


def fully_finetune_bert():
    model_name = "bert-base-multilingual-cased"
    model = AutoModelForTokenClassification.from_pretrained(
        model_name, num_labels=9, id2label=id2label, label2id=label2id
    )

    training_args = TrainingArguments(
        output_dir=f"ner_checkpoints/ner_fully_finetuned_{model_name}",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit = 2,
        load_best_model_at_end=True,
    )

    dataset = {"train": train_dataset, "val": val_dataset, "test": test_dataset}
    model, metrics, elapsed_training =  fine_tune_model(model_name, dataset, model, training_args=training_args, finetune_technique="Fully-FT")
    current_time = time.strftime("%Y-%m-%d-%H-%M-%S")
    with open("results_fully_finetune_bert.csv", "a") as f:
        f.write(
            f"{current_time},{model_name},SSJ500-NER, {metrics},{elapsed_training}\n"
        )
    print(f"Training time fully_finetune_bert: {elapsed_training}")

fully_finetune_bert()


# In[ ]:


def run_lora_sloberta(dataset):
    model_name = "EMBEDDIA/sloberta"
    task_type = TaskType.TOKEN_CLS
    training_args = TrainingArguments(
        output_dir=f"ner_checkpoints/ner_lora_finetuned_{model_name}",
        learning_rate=1e-3,
        per_device_train_batch_size=24,
        per_device_eval_batch_size=24,
        num_train_epochs=3,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit = 2,
        load_best_model_at_end=True,
    )
    model = AutoModelForTokenClassification.from_pretrained(
        model_name, num_labels=9, id2label=id2label, label2id=label2id
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
        task_type=TaskType.TOKEN_CLS, inference_mode=False, r=16, lora_alpha=16, lora_dropout=0.1, bias="all"
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
            f"{current_time},{model_name},SSJ500-NER, {metrics},{elapsed_training}\n"
        )


# In[ ]:


run_lora_sloberta(dataset = {"train": train_dataset, "val": val_dataset, "test": test_dataset})


# In[ ]:


def run_lora_bert(dataset):
    model_name = "bert-base-multilingual-cased"
    task_type = TaskType.TOKEN_CLS
    training_args = TrainingArguments(
        output_dir=f"ner_checkpoints/ner_lora_finetuned_{model_name}",
        learning_rate=1e-3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit = 2,
        load_best_model_at_end=True,
    )
    model = AutoModelForTokenClassification.from_pretrained(
        model_name, num_labels=9, id2label=id2label, label2id=label2id
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
        task_type=TaskType.TOKEN_CLS, inference_mode=False, r=16, lora_alpha=16, lora_dropout=0.1, bias="all"
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
    print(f"Training time run_lora_bert: {elapsed_training}")
    current_time = time.strftime("%Y-%m-%d-%H-%M-%S")
    with open("results_run_lora_bert.csv", "a") as f:
        f.write(
            f"{current_time},{model_name},SSJ500-NER, {metrics},{elapsed_training}\n"
        )


# In[ ]:


run_lora_bert(dataset = {"train": train_dataset, "val": val_dataset, "test": test_dataset})


# In[ ]:


def run_prefix_tune_sloberta(dataset):
    model_name = "EMBEDDIA/sloberta"

    task_type = TaskType.TOKEN_CLS
    training_args = TrainingArguments(
        output_dir=f"ner_checkpoints/ner_prefix_tunning_finetuned_{model_name}",
        learning_rate=1e-2,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit = 2,
        load_best_model_at_end=True,
    )
    model = AutoModelForTokenClassification.from_pretrained(
        model_name, num_labels=9, id2label=id2label, label2id=label2id
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

    prefix_config = PrefixTuningConfig(task_type="TOKEN_CLS", num_virtual_tokens=20)

    model = get_peft_model(model, prefix_config)

    _, metrics, elapsed_training = fine_tune_model(
        model_name, dataset, model, training_args, finetune_technique="P-tunning"
    )
    print(f"Training time run_prefix_tune_sloberta: {elapsed_training}")
    current_time = time.strftime("%Y-%m-%d-%H-%M-%S")
    with open("results_run_prefix_tune_sloberta.csv", "a") as f:
        f.write(
            f"{current_time},{model_name},SSJ500-NER, {metrics},{elapsed_training}\n"
        )


# In[ ]:


run_prefix_tune_sloberta(dataset = {"train": train_dataset, "val": val_dataset, "test": test_dataset})


# In[ ]:


def run_prefix_tune_bert(dataset):
    model_name = "bert-base-multilingual-cased"

    task_type = TaskType.TOKEN_CLS
    training_args = TrainingArguments(
        output_dir=f"ner_checkpoints/ner_prefix_tunning_finetuned_{model_name}",
        learning_rate=1e-2,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit = 2,
        load_best_model_at_end=True,
    )
    model = AutoModelForTokenClassification.from_pretrained(
        model_name, num_labels=9, id2label=id2label, label2id=label2id
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

    prefix_config = PrefixTuningConfig(task_type="TOKEN_CLS", num_virtual_tokens=20)

    model = get_peft_model(model, prefix_config)

    _, metrics, elapsed_training = fine_tune_model(
        model_name, dataset, model, training_args, finetune_technique="P-tunning"
    )
    print(f"Training time run_prefix_tune_bert: {elapsed_training}")
    current_time = time.strftime("%Y-%m-%d-%H-%M-%S")
    with open("results_run_prefix_tune_bert.csv", "a") as f:
        f.write(
            f"{current_time},{model_name},SSJ500-NER, {metrics},{elapsed_training}\n"
        )


# In[ ]:


run_prefix_tune_bert(dataset = {"train": train_dataset, "val": val_dataset, "test": test_dataset})


# In[ ]:




def run_ia3_sloberta(dataset):
    model_name = "EMBEDDIA/sloberta"
    task_type = TaskType.TOKEN_CLS
    training_args = TrainingArguments(
        output_dir=f"ner_checkpoints/ner-ia3-{model_name}",
        learning_rate=1e-2,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit = 2,
        load_best_model_at_end=True,
    )
    model = AutoModelForTokenClassification.from_pretrained(
        model_name, num_labels=9, id2label=id2label, label2id=label2id
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
        model_name, dataset, model, training_args, finetune_technique="IA3"
    )
    print(f"Training time: {elapsed_training}")
    current_time = time.strftime("%Y-%m-%d-%H-%M-%S")
    with open("results_run_ia3_sloberta.csv", "a") as f:
        f.write(
            f"{current_time},{model_name},SSJ500-NER, {metrics},{elapsed_training}\n"
        )


# In[ ]:


run_ia3_sloberta(dataset = {"train": train_dataset, "val": val_dataset, "test": test_dataset})


# In[ ]:


def run_ia3_bert(dataset):
    model_name = "bert-base-multilingual-cased"
    task_type = TaskType.TOKEN_CLS
    training_args = TrainingArguments(
        output_dir=f"ner_checkpoints/ner-ia3-{model_name}",
        learning_rate=1e-1,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit = 2,
        load_best_model_at_end=True,
    )
    model = AutoModelForTokenClassification.from_pretrained(
        model_name, num_labels=9, id2label=id2label, label2id=label2id
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
            f"{current_time},{model_name},NER, {metrics},{elapsed_training}\n"
        )


# In[ ]:


run_ia3_bert(dataset = {"train": train_dataset, "val": val_dataset, "test": test_dataset})



def run_bitfit_sloberta(dataset):
    model_name = "EMBEDDIA/sloberta"
    task_type = TaskType.TOKEN_CLS
    training_args = TrainingArguments(
        output_dir=f"ner_checkpoints/ner_bitfit_finetuned_{model_name}",
        learning_rate=1e-3,
        per_device_train_batch_size=24,
        per_device_eval_batch_size=24,
        num_train_epochs=3,
        weight_decay=0.0001,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit = 2,
        load_best_model_at_end=True,
    )
    model = AutoModelForTokenClassification.from_pretrained(
        model_name, num_labels=9, id2label=id2label, label2id=label2id
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
            f"{current_time},{model_name},SSJ500-NER, {metrics},{elapsed_training}\n"
        )


run_bitfit_sloberta(dataset = {"train": train_dataset, "val": val_dataset, "test": test_dataset})


def run_bitfit_bert(dataset):
    model_name = "bert-base-multilingual-cased"
    task_type = TaskType.TOKEN_CLS
    training_args = TrainingArguments(
        output_dir=f"ner_checkpoints/ner_bitfit_finetuned_{model_name}",
        learning_rate=1e-4,
        per_device_train_batch_size=24,
        per_device_eval_batch_size=24,
        num_train_epochs=3,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit = 2,
        load_best_model_at_end=True,
    )
    model = AutoModelForTokenClassification.from_pretrained(
        model_name, num_labels=9, id2label=id2label, label2id=label2id
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
            f"{current_time},{model_name},SSJ500-NER, {metrics},{elapsed_training}\n"
        )

run_bitfit_bert(dataset = {"train": train_dataset, "val": val_dataset, "test": test_dataset})