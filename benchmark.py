import csv
import os
import time
import numpy as np
import pandas as pd
import wandb
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from peft import get_peft_model, LoraConfig, TaskType, PrefixTuningConfig

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


# Constants

WANDB_API_KEY = ""
DATASET_PATH = "SuperGLUE-GoogleMT/csv"
OUTPUT_CSV = "results.csv"


class GLUEBenchmark:
    """Benchmark for the SuperGLUE tasks.

    This class provides methods for computing metrics, loading datasets, models, and tokenizers,
    initializing wandb, training and evaluating models, and writing results to a CSV file.
    It supports two benchmark tasks: Boolean questions (BoolQ) and Causal-based (CB).
    Supported PEFT types are LORA and PrefixTune.

    Attributes:
        model_name (str): The name of the model used for the benchmark.

    Methods:
        compute_metrics_binary: Compute metrics for binary classification tasks.
        compute_metrics_multiclass: Compute metrics for multiclass classification tasks.
        softmax: Compute the softmax of a vector x.
        make_wandb_config: Make a config dictionary for wandb.
        wandb_init: Initialize wandb.
        load_dataset: Load and preprocess the task dataset.
        load_model: Load the model for the task.
        load_tokenizer: Load the tokenizer for the task.
        load_peft_model: Load the PEFT model for the task.
        get_trainer: Get the trainer for the task.
        write_results: Write the results to a CSV file.
        boolq: Boolean questions benchmark.
        cb: Causal-based benchmark.
    """


class GLUEBenchmark:
    """Benchmark for the SuperGLUE tasks."""

    def compute_metrics_binary(self, eval_pred):
        """Compute metrics for binary classification tasks."""
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average="binary"
        )
        return {
            "accuracy": accuracy,
            "f1": f1,
            "precision": precision,
            "recall": recall,
        }

    def compute_metrics_multiclass(self, eval_pred):
        """Compute metrics for the multiclass classification task."""
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average="weighted"
        )
        return {
            "accuracy": accuracy,
            "f1": f1,
            "precision": precision,
            "recall": recall,
        }

    def softmax(self, x):
        """Compute the softmax of a vector x."""
        return np.exp(x) / np.sum(np.exp(x), axis=0)

    def __init__(self, model_name):
        self.model_name = model_name

    def make_wandb_config(
        self, model_name, task_name, batch_size, learning_rate, num_epochs
    ):
        """Make a config dictionary for wandb."""
        return {
            "model_name": model_name,
            "task_name": task_name,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "num_epochs": num_epochs,
        }

    def wandb_init(self, config={}):
        """Initialize wandb."""
        if WANDB_API_KEY != "":
            wandb.login(key=WANDB_API_KEY)
            wandb.init(project="peft", config=config)

    def load_dataset(self, task_name):
        """Load and preprocess the task dataset."""
        try:
            train = Dataset.from_pandas(
                pd.read_csv(f"{DATASET_PATH}/{task_name}/train.csv").drop("idx", axis=1)
            )
            val = Dataset.from_pandas(
                pd.read_csv(f"{DATASET_PATH}/{task_name}/val.csv").drop("idx", axis=1)
            )
            test = Dataset.from_pandas(
                pd.read_csv(f"{DATASET_PATH}/{task_name}/test.csv").drop("idx", axis=1)
            )
            return {"train": train, "validation": val, "test": test}
        except FileNotFoundError:
            raise ValueError("Task not supported.")

    def load_model(self, task_name):
        """Load the model for the task."""
        if task_name == "BoolQ":
            return AutoModelForSequenceClassification.from_pretrained(
                self.model_name, num_labels=2
            )
        elif task_name == "CB":
            return AutoModelForSequenceClassification.from_pretrained(
                self.model_name, num_labels=3
            )
        else:
            raise ValueError("Task not supported.")

    def load_tokenizer(self, task_name):
        """Load the tokenizer for the task."""
        if task_name == "BoolQ":
            return AutoTokenizer.from_pretrained(self.model_name)
        elif task_name == "CB":
            return AutoTokenizer.from_pretrained(self.model_name)
        else:
            raise ValueError("Task not supported.")

    def load_peft_model(self, task_name, model, peft_type="lora"):
        """Load the PEFT model for the task."""
        if task_name in ["BoolQ", "CB"]:
            if peft_type == "lora":
                if self.model_name == "EMBEDDIA/sloberta":
                    target_modules = [
                        f"roberta.encoder.layer.{i}.attention.self.{part}"
                        for i in range(model.config.num_hidden_layers)
                        for part in ["query", "key"]
                    ]
                peft_config = LoraConfig(
                    task_type=TaskType.SEQ_CLS,
                    r=16,
                    lora_alpha=16,
                    lora_dropout=0.1,
                    bias="all",
                    target_modules=target_modules,
                )
                return get_peft_model(model, peft_config)
            elif peft_type == "prefixtune":
                peft_config = PrefixTuningConfig(
                    task_type=TaskType.SEQ_CLS,
                    num_virtual_tokens=20,
                )
                return get_peft_model(model, peft_config)
            else:
                raise ValueError("PEFT type not supported.")

    def get_trainer(
        self, peft_model, train, val, task_name, model_name, peft_type, compute_metrics
    ):
        """Get the trainer for the task."""

        training_args = TrainingArguments(
            output_dir=f"results/{model_name}/{task_name}/{peft_type}",
            evaluation_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            num_train_epochs=3,
            weight_decay=0.01,
        )

        trainer = Trainer(
            model=peft_model,
            args=training_args,
            train_dataset=train,
            eval_dataset=val,
            compute_metrics=compute_metrics,
        )

        return trainer

    def write_results(
        self,
        model_name,
        task_name,
        test_results,
        eval_results,
        peft_type,
        test,
        train_time,
    ):
        """Write the results to a CSV file."""

        print(eval_results)

        with open(OUTPUT_CSV, mode="a") as f:
            writer = csv.writer(f)
            if f.tell() == 0:
                writer.writerow(
                    [
                        "model_name",
                        "task_name",
                        "eval_loss",
                        "eval_accuracy",
                        "eval_f1",
                        "eval_precision",
                        "eval_recall",
                        "eval_runtime",
                        "eval_samples_per_second",
                        "eval_steps_per_second",
                        "epoch",
                        "train_time",
                    ]
                )
            writer.writerow(
                [
                    model_name,
                    task_name,
                    eval_results["eval_loss"],
                    eval_results["eval_accuracy"],
                    eval_results["eval_f1"],
                    eval_results["eval_precision"],
                    eval_results["eval_recall"],
                    eval_results["eval_runtime"],
                    eval_results["eval_samples_per_second"],
                    eval_results["eval_steps_per_second"],
                    eval_results["epoch"],
                    train_time,
                ]
            )

    def boolq(self, batch_size=32, learning_rate=2e-5, num_epochs=3, peft_type="lora"):
        """Boolean questions benchmark."""
        task_name = "BoolQ"
        model_name = self.model_name
        batch_size = batch_size
        learning_rate = learning_rate
        num_epochs = num_epochs

        config = self.make_wandb_config(
            model_name, task_name, batch_size, learning_rate, num_epochs
        )
        self.wandb_init(config)

        train, val, test = self.load_dataset(task_name).values()

        tokenizer = self.load_tokenizer(task_name)
        model = self.load_model(task_name)
        peft_model = self.load_peft_model(task_name, model, peft_type=peft_type)

        def tokenize_BoolQ(examples):
            return tokenizer(
                examples["passage"],
                examples["question"],
                padding="max_length",
                truncation=True,
                max_length=512,
            )

        train, val, test = (
            train.map(tokenize_BoolQ, batched=True),
            val.map(tokenize_BoolQ, batched=True),
            test.map(tokenize_BoolQ, batched=True),
        )
        trainer = self.get_trainer(
            peft_model,
            train,
            val,
            task_name,
            model_name,
            peft_type,
            self.compute_metrics_binary,
        )

        trainer.train()
        eval_results = trainer.evaluate()
        if WANDB_API_KEY != "":
            wandb.log(eval_results)

        test_results = trainer.predict(test)
        self.write_results(model_name, task_name, test_results, peft_type, test)

        out_file = f"results_{model_name}_{task_name}_{peft_type}.csv"

        directory = os.path.dirname(out_file)
        if not os.path.exists(directory):
            os.makedirs(directory)

        with open(out_file, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["passage", "question", "prediction"])
            for i in range(len(test)):
                writer.writerow(
                    [
                        test["passage"][i],
                        test["question"][i],
                        test_results.predictions[i],
                    ]
                )

    def cb(self, batch_size=32, learning_rate=2e-5, num_epochs=3, peft_type="lora"):
        """Causal-based benchmark."""
        task_name = "CB"
        model_name = self.model_name
        batch_size = batch_size
        learning_rate = learning_rate
        num_epochs = num_epochs

        config = self.make_wandb_config(
            model_name, task_name, batch_size, learning_rate, num_epochs
        )
        self.wandb_init(config)

        train, val, test = self.load_dataset(task_name).values()

        tokenizer = self.load_tokenizer(task_name)
        model = self.load_model(task_name)
        peft_model = self.load_peft_model(task_name, model, peft_type=peft_type)

        def tokenize_CB(examples):
            return tokenizer(
                examples["premise"],
                examples["hypothesis"],
                padding="max_length",
                truncation=True,
                max_length=512,
            )

        def encode_labels(examples):
            label_dict = {"entailment": 0, "neutral": 1, "contradiction": 2}
            # Replace labels in the examples with encoded labels
            examples["label"] = [
                label_dict[label] for label in examples["label"]
            ].copy()
            return examples

        train = train.map(encode_labels, batched=True)
        val = val.map(encode_labels, batched=True)

        train = train.map(tokenize_CB, batched=True)
        val = val.map(tokenize_CB, batched=True)
        test = test.map(tokenize_CB, batched=True)

        trainer = self.get_trainer(
            peft_model,
            train,
            val,
            task_name,
            model_name,
            peft_type,
            self.compute_metrics_multiclass,
        )

        start = time.time()
        trainer.train()
        elapsed = time.time() - start

        eval_results = trainer.evaluate()
        if WANDB_API_KEY != "":
            wandb.log(eval_results)

        test_results = trainer.predict(test)
        self.write_results(
            model_name, task_name, test_results, eval_results, peft_type, test, elapsed
        )

        out_file = f"results_{model_name}_{task_name}_{peft_type}.csv"

        directory = os.path.dirname(out_file)
        if not os.path.exists(directory):
            os.makedirs(directory)

        with open(out_file, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["premise", "hypothesis", "prediction"])
            label_dict = {0: "entailment", 1: "neutral", 2: "contradiction"}
            for i in range(len(test)):
                # Write each result row
                writer.writerow(
                    [
                        test["premise"][i],
                        test["hypothesis"][i],
                        label_dict[self.softmax(test_results.predictions[i]).argmax()],
                    ]
                )

    def copa(self, batch_size=32, learning_rate=2e-5, num_epochs=3, peft_type="lora"):
        """Benchmark for the Choice of Plausible Alternatives task."""
        task_name = "COPA"
        pass  # TODO

    def multirc(
        self, batch_size=32, learning_rate=2e-5, num_epochs=3, peft_type="lora"
    ):
        """Benchmark for the MultiRC task."""
        task_name = "MultiRC"
        pass  # TODO

    def record(self, batch_size=32, learning_rate=2e-5, num_epochs=3, peft_type="lora"):
        """Benchmark for the ReCoRD task."""
        task_name = "ReCoRD"
        pass  # TODO

    def rte(self, batch_size=32, learning_rate=2e-5, num_epochs=3, peft_type="lora"):
        """Benchmark for the Recognizing Textual Entailment task."""
        task_name = "RTE"
        pass  # TODO

    def wsc(self, batch_size=32, learning_rate=2e-5, num_epochs=3, peft_type="lora"):
        """Benchmark for the Winograd Schema Challenge task."""
        task_name = "WSC"
        pass  # TODO


if __name__ == "__main__":
    benchmark = GLUEBenchmark("EMBEDDIA/sloberta")
    benchmark.cb(num_epochs=10, peft_type="prefixtune")