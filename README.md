# RADCoT
(official) Code for "RADCoT: Retrieval-Augmented Distillation to Specialization Models for Generating Chain-of-Thoughts in Query Expansion", LREC-COLING 2024 (accepted)

## Requirements
* [PyTorch](http://pytorch.org/) >= 1.9
* numpy
* faiss-cpu
* transformers==3.0.2
* tensorboard

## Process

1. Environment Setting
```console
pip install -r ./pretrain/requirements.txt
```

2. Chain-of-Thought generation (using LLM)
```console
cd llm_infer
python infer.py
```

3. Retrieval-augmented SLM training and inference

```console
bash train.sh 4
bash infer.sh
```

4. evaluation

```console
python retrieve_bm25.py --eval_dataset trec-dl-19 --query_augment_file_path llm_infer/results/19_bm25.txt
```


## References
* [FiD](https://github.com/facebookresearch/FiD)
* [Query2Doc](https://aclanthology.org/2023.emnlp-main.585/)

## Q&A
If you encounter any problem, leave an issue in the github repo.
