import argparse
import torch
import time
import pandas as pd
import numpy as np
import os
import itertools
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    EarlyStoppingCallback, 
    TrainerState, 
    TrainerControl,
    set_seed
)
from sklearn.model_selection import train_test_split
from datasets import Dataset
from datasets import load_dataset, load_metric
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType, PrefixTuningConfig, IA3Config
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
os.environ["WANDB_DISABLED"] = "true"

model_name = "bert-base-multilingual-cased"
# Load the dataset
dataset = load_dataset('csv', data_files={
    'train': 'RTE/train.csv',
    'validation': 'RTE/val.csv'
})

# Tokenization
tokenizer = AutoTokenizer.from_pretrained(model_name)
num_virtual_tokens = 20  # Number of virtual tokens used in prefix tuning
max_length = 512 - num_virtual_tokens
def preprocess_function(examples):
    return tokenizer(examples['premise'], examples['hypothesis'], truncation=True, padding='max_length', max_length=max_length)
dataset = dataset.map(preprocess_function, batched=True)

# Here you split the validation set into validation and test sets
test_train_split = dataset['validation'].train_test_split(test_size=0.5)
# Now you need to add these new sets back into your dataset
dataset['validation'] = test_train_split['train']
dataset['test'] = test_train_split['test']

# Now continue with label mapping
def label_mapping(example):
    label_dict = {'not_entailment': 0, 'entailment': 1}
    example['labels'] = label_dict[example['label']]
    return example
dataset = dataset.map(label_mapping)

# Set format for PyTorch
dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
class MemoryUsageCallback(TrainerCallback):
    def __init__(self, log_file):
        self.log_file = log_file
        self.memory_usage = []

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if torch.cuda.is_available():
            allocated_memory = torch.cuda.memory_allocated()
            reserved_memory = torch.cuda.memory_reserved()
            log_message = f"Step {state.global_step} - Allocated Memory: {allocated_memory}, Reserved Memory: {reserved_memory}"
            print(log_message)
            with open(self.log_file, "a") as log_f:
                log_f.write(log_message + "\n")
            self.memory_usage.append((allocated_memory, reserved_memory))

    def get_memory_usage(self):
        return self.memory_usage

    def get_max_memory_usage_percentage(self):
        if not self.memory_usage:
            return 0
        max_percentage = max(allocated / reserved * 100 for allocated, reserved in self.memory_usage if reserved > 0)
        return max_percentage

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    f1 = f1_score(labels, predictions, average='weighted')  # Modify as needed
    accuracy = accuracy_score(labels, predictions)
    return {
        'f1': f1,
        'accuracy': accuracy
    }

def fine_tune_model(model_name, model, training_args, dataset, patience, threshold):
    memory_callback = MemoryUsageCallback(log_file=f"{model_name}_memory_usage.log")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=patience, early_stopping_threshold=threshold), memory_callback]
    )

    start = time.time()
    trainer.train()
    elapsed_training = time.time() - start

    metrics = trainer.evaluate(dataset['test'])

    print(f"model: {model_name}, Dataset: Sentinews, Test Metrics: {metrics}")

    model.save_pretrained(f"{model_name}_RTE_FINAL")
    memory_usage = memory_callback.get_memory_usage()
    max_memory_usage_percentage = memory_callback.get_max_memory_usage_percentage()

    return model, metrics, elapsed_training, max_memory_usage_percentage

def count_trainable_parameters(model):
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {trainable_params}")



def run_prefix_tune_sloberta(dataset, patience, threshold, model_name="EMBEDDIA/sloberta"):
    task_type = TaskType.SEQ_CLS
    training_args = TrainingArguments(
        output_dir=f"{model_name}-sentinews",
        learning_rate=1e-4,
        per_device_train_batch_size=24,
        per_device_eval_batch_size=24,
        num_train_epochs=15,
        weight_decay=0.1,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
    model = prepare_model_for_kbit_training(model, task_type)


    prefix_config = PrefixTuningConfig(task_type="SEQ_CLS", num_virtual_tokens=20)

    model = get_peft_model(model, prefix_config)
    print('Prefix Tune for ', model_name)
    count_trainable_parameters(model)
    _, metrics, elapsed_training, memory_usage = fine_tune_model(
        model_name, model, training_args, dataset, patience, threshold
    )

    current_time = time.strftime("%Y-%m-%d-%H-%M-%S")
    with open("results.csv", "a") as f:
        f.write(
            f"{current_time},{model_name},RTE-prefix_tune,{metrics},{elapsed_training},{memory_usage},{patience}, {threshold}\n"
        )

def run_ia3_sloberta(dataset, patience, threshold, model_name="EMBEDDIA/sloberta"):
    task_type = TaskType.SEQ_CLS
    training_args = TrainingArguments(
        output_dir=f"{model_name}-sentinews",
        learning_rate=1e-4,
        per_device_train_batch_size=24,
        per_device_eval_batch_size=24,
        num_train_epochs=15,
        weight_decay=0.1,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
    model = prepare_model_for_kbit_training(model, task_type)

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
    ia3_config = IA3Config(task_type=task_type, target_modules=target_modules, feedforward_modules=target_modules)

    model = get_peft_model(model, ia3_config)
    print('ia3 for ', model_name)
    count_trainable_parameters(model)
    _, metrics, elapsed_training, memory_usage = fine_tune_model(
        model_name, model, training_args, dataset, patience, threshold
    )

    current_time = time.strftime("%Y-%m-%d-%H-%M-%S")
    with open("results.csv", "a") as f:
        f.write(
            f"{current_time},{model_name},RTE-ia3,{metrics},{elapsed_training},{memory_usage},{patience}, {threshold}\n"
        )

def prepare_model_for_bitfit(model):
    # Freeze all parameters except for the biases
    for name, param in model.named_parameters():
        if 'bias' not in name:
            param.requires_grad = False
    return model

def run_bitfit_sloberta(dataset, patience, threshold, model_name="EMBEDDIA/sloberta"):
    task_type = TaskType.SEQ_CLS
    training_args = TrainingArguments(
        output_dir=f"{model_name}-sentinews",
        learning_rate=1e-4,
        per_device_train_batch_size=24,
        per_device_eval_batch_size=24,
        num_train_epochs=15,
        weight_decay=0.1,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
    model = prepare_model_for_bitfit(model)
    print('BitFit for ', model_name)
    count_trainable_parameters(model)
    _, metrics, elapsed_training, memory_usage = fine_tune_model(
        model_name, model, training_args, dataset, patience, threshold
    )

    current_time = time.strftime("%Y-%m-%d-%H-%M-%S")
    with open("results.csv", "a") as f:
        f.write(
            f"{current_time},{model_name},RTE-bitfit,{metrics},{elapsed_training},{memory_usage},{patience}, {threshold}\n"
        )

def run_lora_sloberta(dataset, patience, threshold, model_name="EMBEDDIA/sloberta"):
    task_type = TaskType.SEQ_CLS  # You might need a different TaskType depending on your exact use case

    training_args = TrainingArguments(
        output_dir=f"{model_name}-RTE",  # Change as needed
        learning_rate=1e-4,
        per_device_train_batch_size=16,  # Adjust based on your GPU memory
        per_device_eval_batch_size=16,
        num_train_epochs=15,
        weight_decay=0.1,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True
    )

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)  # Adjust num_labels as needed
    model = prepare_model_for_kbit_training(model, task_type)

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
    
    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        lora_dropout=0.05,
        task_type=task_type,
        bias="none",
        target_modules=target_modules,
    )

    model = get_peft_model(model, lora_config)
    print_trainable_parameters(model)
    
    _, metrics, elapsed_training, memory_usage = fine_tune_model(
        model_name, model, training_args, dataset, patience, threshold
    )

    current_time = time.strftime("%Y-%m-%d-%H-%M-%S")
    with open("results.csv", "a") as f:
        f.write(
            f"{current_time},{model_name},RTE-lora,{metrics},{elapsed_training},{memory_usage},{patience}, {threshold}\n"
        )
def run_fully_finetune(dataset, patience, threshold, model_name="EMBEDDIA/sloberta"):
    task_type = TaskType.SEQ_CLS  # You might need a different TaskType depending on your exact use case

    training_args = TrainingArguments(
        output_dir=f"{model_name}-RTE",  # Change as needed
        learning_rate=1e-4,
        per_device_train_batch_size=16,  # Adjust based on your GPU memory
        per_device_eval_batch_size=16,
        num_train_epochs=15,
        weight_decay=0.1,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True
    )

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)  # Adjust num_labels as needed
    
    _, metrics, elapsed_training, memory_usage = fine_tune_model(
        model_name, model, training_args, dataset, patience, threshold
    )

    current_time = time.strftime("%Y-%m-%d-%H-%M-%S")
    with open("results.csv", "a") as f:
        f.write(
            f"{current_time},{model_name},RTE-FFT,{metrics},{elapsed_training},{memory_usage},{patience}, {threshold}\n"
        )
# Function to run experiments with a given seed
def run_experiments_with_seed(seed, dataset, model_name, model_type):
    # Set the seed for reproducibility
    set_seed(seed)
    
    # Run the specified experiment
    if model_type == "lora":
        run_lora_sloberta(dataset, patience=1, threshold=0.01, model_name=model_name)
    elif model_type == "prefix_tune":
        run_prefix_tune_sloberta(dataset, patience=3, threshold=0.1, model_name=model_name)
    elif model_type == "ia3":
        run_ia3_sloberta(dataset, patience=6, threshold=0.01, model_name=model_name)
    elif model_type == "bitfit":
        run_bitfit_sloberta(dataset, patience=1, threshold=0.001, model_name=model_name)
    elif model_type == "fft":
        run_fully_finetune(dataset, patience=1, threshold=0.001, model_name=model_name)
    else:
        raise ValueError(f"Unknown model type: {model_type}")



# Argument parsing
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run fine-tuning experiments with different model types")
    parser.add_argument("--model_type", type=str, required=True, choices=["lora", "prefix_tune", "ia3", "bitfit","fft"], help="Type of model to run")
    args = parser.parse_args()

    seeds = [42, 123, 456, 789, 1000]  # List of seeds you want to use
    model_name = "bert-base-multilingual-cased"
    for seed in seeds:
        run_experiments_with_seed(seed, dataset, model_name, args.model_type)
