# Structure
```
.
├── Dockerfile
├── LICENSE
├── README.md
├── infer.py - inference of model
├── requirements.txt
├── start.sh - perform this to compile Docker image and launch a container
├── train.py - model training. Perform it before launching infer.py
└── utils
    ├── __init__.py
    ├── correlation_analysis.py — crosslayer similarity metrics
    ├── evaluation.py — sidekick functions for getting predictions etc
    ├── load_dataset.py — methods for loading data from sources and processing them into a format suitable for training/inference
    └── metrics.py — metrics for evaluating the model's performance

```

# Language complexity in multilingual language models

This repo contains files of experiments related to my thesis project.

Now the work is not completed, it will be supplemented as the tasks progress and results are achieved.

In addition, this repository is a reporting project for the course MLOps (thanks for the [template](https://github.com/v-goncharenko/data-science-template/tree/master))

## Problem formulation

To study the hidden states of BERT when transferring learning between languages (both of the same language group and different ones)

This work will help to gain more knowledge about BERT's hidden states and explore the connections between the model prediction process in different languages. This knowledge is useful for transferring learning to low-resource languages.

## Tasks

1. Fine-tuning BERT classifier on english articles from Vikidia (easy texts) and Wikiedia (normal texts). 
2. Investigating BERT's hidden states using [ANC](https://github.com/TartuNLP/xsim/tree/master?tab=readme-ov-file) and other metrics on prediction for french, italian, spanish, deutsche and russian languages.
3. Fine-tuning BERT classifier on [ReadMe++](https://github.com/tareknaous/readme). 
4. Investigate [ANC](https://github.com/TartuNLP/xsim/tree/master?tab=readme-ov-file) on each class.
5. Compare metrics on our task and on [XNLI](https://aclanthology.org/2022.aacl-main.15/) (consider other tasks if possible)


## Data
1. [Vikidia](https://ssharoff.github.io/kelly/dfly/) (Wikipedia for children) and Wikipedia. There are articles with same title written on Wikipedia and rewritten (by human) in simpl words for Vikidia. We will assume them to be different in language complexity. Advantage: lots of datum, disadvantage: only 2 classes.
2. [ReadMe++](https://github.com/tareknaous/readme) awesome dataset in 5 languages, divided into 6 classes using CEFR. The main disadvantage is the uneven number of entries in classes. 

## Modeling approach

We will use BERT `bert-base-multilingual-cased` (from `transformers`, more details on [huggingface](https://huggingface.co/google-bert/bert-base-multilingual-cased)) since there are results in this [article](https://aclanthology.org/2022.aacl-main.15/) for XNLI task. After obtaining significant results we may enlarge work on other models.

For all classification tasks we will use ready-made `torch` optimizers and losses.

## Prediction method

For successful prediction, we need to load weights into the model, apply a tokenizer to the input data and feed them to the model input. As output we well receive predictions and and tensor of all hidden states. If the task was to obtain language complexity for some batch of text we will load predictions into file. If the task was to compare hidden states on reference language and on prediction we will need to apply metric (for example ANC) on output of hidden states and on saved previously reference hidden states.