# Natural language processing course 2023/24: Parameter-Efficient Fine-Tuning of Language Models

This project focuses on investigating parameter-efficient techniques for fine-tuning large language models, such as Low-Rank Adaptation (LoRA), soft prompts, etc. We compare different approaches across various NLP tasks to assess the efficiency and effectiveness of each fine-tuning strategy. The evaluation considers performance metrics such as precision and f1 scores, computational efficiency (memory used and runtime). The data consists of a set of benchmark of several NLP task (these are: Natural Language Interaction, Name Entity Recognition, Dependency Parsing and Recognition of Entailment) in the slovene languge obtained from the slobench evaluation framework (https://slobench.cjvt.si/).

## Contents

### Benchmarks

This are the jupyter notebooks containing the code to perform the experiments:

-   lora_RTE_BERT_multilingual.ipynb, lora_RTE_sloberta.ipynb, lora_rte_BMU.py, lora_rte_sloberta.ipynb.py. : Textual Entailment Recognition Benchmarks
-   ner_benchmark.ipynb/.py: Name Entity Recognition benchmarks
-   dependency-parsing_benchmark.ipynb/.py: Dependency Parsing Relation Detection benchmark
-   prefix_tuning_BoolIQ.ipynb: Natural Language Understanding benchmark
-   sentiment_benchmark.ipynb/.py: Sentiment Analysis benchmark
-   nli_benchmark.py: Natural Language Interaction benchmark
-   run_container.sh: Bash script to run a bechmark script several times in a SLURM cluster and record the results.
-   run_container_rte.sh: Bash script to run the RTE task benchmarks in a SLURM cluster, which have a different interface than the rest of benchmarks.

### Configuration files

-   sentiment_analysis.def: Apptainer image definition file to upload the sentiment analysis benchmark to an Apptainer enabled cluster.
-   conda_base.yaml: Definition file of conda environment used in exeperiments for replicability in the different experimental setups used.
-   requirement.txt: In case conda is not available to you, you can use this requirement file to install all dependencies needed to run the project directly into a virtual environment.

### Report

- The LaTEX source code for the report that summarizes all results obtained can be seen in the **report** directory. In it, NLP_Report.pdf is the final result of the compiled source. The main code for the report is contained in report.tex.


### Installation:

#### Locally

To install dependencies you can just use pip:

```
pip install -r requirements.txt
```

Although pip is our preferred package manager for this work, you can also use conda. To do so, use the following command:

```
conda env create -f conda_base.yaml
```

#### Container image

Assuming you have a working apptainer or singularity installation, to build the project's container and install all dependencies use the following command:

```
mkdir containers
apptainer build containers/nlp_benchmark.sif  sentiment_analysis.def
```

This is the recommended way of installing the experimental environment for reproducible results.

### Usage

Our training scripts are contained in the .py scripts. To run a benchmark, use the following command in a SLURM cluster (the script uses internally a nlp_benchmark apptainer image, so you need to build it before executing the script):

```
sbatch run_container.sh [name of benchmark].py
```

For instance, to run the ner_benchmark use the following command:

```
sbatch run_container.sh ner_benchmark.py
```

You can also run code localy like any other python script assuming you have dependencies installed. To run the the ner_benchmark just use:

```
python ner_benchmark.py
```
