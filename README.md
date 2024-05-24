# Natural language processing course 2023/24: Parameter-Efficient Fine-Tuning of Language Models

This project focuses on investigating parameter-efficient techniques for fine-tuning large language models, such as Low-Rank Adaptation (LoRA), soft prompts, etc. We compare different approaches across various NLP tasks to assess the efficiency and effectiveness of each fine-tuning strategy. The evaluation will consider model performance, computational efficiency, and adaptability to different tasks. The data used is mainly slovenian corpus sourced from the slobench evaluation framework (https://slobench.cjvt.si/).

## Contents

### Benchmarks

This are the jupyter notebooks containing the code to perform the experiments:

-   lora_RTE_BERT_multilingual.ipynb, lora_RTE_BMU.py, lora_RTE_sloberta.ipynb.py. : Textual Entailment Recognition Benchmarks
-   ner_benchmark.ipynb/.py: Name Entity Recognition benchmarks
-   dependency-parsing_benchmark.ipynb/.py: Dependency Parsing Relation Detection benchmark
-   prefix_tuning_BoolIQ.ipynb: Natural Language Understanding benchmark
-   sentiment_benchmark.ipynb/.py: Sentiment Analysis benchmark
-   nli_benchmark.py: Natural Language Interaction benchmark
-   run_container.sh: Bash script to run a bechmark script several times in the cluster and record the results.

### Configuration files

-   sentiment_analysis.def: Apptainer image definition file to upload the sentiment analysis benchmark to an Apptainer enabled cluster.
-   conda_base.yaml: Definition file of conda environment used in exeperiments for replicability in the different experimental setups used.
-   requirement.txt: In case conda is not available to you, you can use this requirement file to install all dependencies needed to run the project directly into a virtual environment.

### Report

- The LaTEX source code for the report that summarizes all results obtained can be seen in the report directory. In it, NLP_Report.pdf is the final result of the compiled source. The main code for the report is contained in report.tex.
