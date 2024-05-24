# Natural language processing course 2023/24: Parameter-Efficient Fine-Tuning of Language Models

This project focuses on investigating parameter-efficient techniques for fine-tuning large language models, such as Low-Rank Adaptation (LoRA), soft prompts, etc. We compare different approaches across various NLP tasks to assess the efficiency and effectiveness of each fine-tuning strategy. The evaluation will consider model performance, computational efficiency, and adaptability to different tasks. The data used is mainly slovenian corpus sourced from the slobench evaluation framework (https://slobench.cjvt.si/).

## Contents

### Benchmarks

This are the jupyter notebooks containing the code to perform the experiments:

-   lora_RTE_BERT_multilingual.ipynb, lora_RTE_sloberta.ipynb: Textual Entailment Recognition Benchmarks
-   lora_coreference.ipynb: Coreference benchmark
-   ner_benchmarkv2.ipynb: Name Entity Recognition benchmarks
-   dependency-parsing_benchmark.ipynb: Dependency Parsing Relation Detection benchmark
-   prefix_tuning_BoolIQ.ipynb: Natural Language Understanding Benchmark
-   sentiment_benchmark.ipynb: Sentiment Analysis benchmark

### Configuration files

-   sentiment_analysis.def: Apptainer image definition file to upload the sentiment analysis benchmark to an Apptainer enabled cluster.

- conda_base.yaml: Definition file of conda environment used in exeperiments for replicability in the different experimental setups used.

### Report

- The LaTEX source code for the report that summarizes all results obtained can be seen in the report directory. In it, NLP_Report.pdf is the final result of the compiled source. The main code for the report is contained in report.tex.
