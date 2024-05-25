# RADCoT
(official) Code for "RADCoT: Retrieval-Augmented Distillation to Specialization Models for Generating Chain-of-Thoughts in Query Expansion", LREC-COLING 2024 (accepted)

## Requirements
* [PyTorch](http://pytorch.org/) >= 1.13.1
* numpy
* faiss-cpu
* transformers==4.33.2
* tensorboard
* pyserini

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



## Citation
```bibtex

@inproceedings{lee-etal-2024-radcot-retrieval,
    title = "{RADC}o{T}: Retrieval-Augmented Distillation to Specialization Models for Generating Chain-of-Thoughts in Query Expansion",
    author = "Lee, Sung-Min  and
      Park, Eunhwan  and
      Jeon, DongHyeon  and
      Kang, Inho  and
      Na, Seung-Hoon",
    editor = "Calzolari, Nicoletta  and
      Kan, Min-Yen  and
      Hoste, Veronique  and
      Lenci, Alessandro  and
      Sakti, Sakriani  and
      Xue, Nianwen",
    booktitle = "Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024)",
    month = may,
    year = "2024",
    address = "Torino, Italia",
    publisher = "ELRA and ICCL",
    url = "https://aclanthology.org/2024.lrec-main.1182",
    pages = "13514--13523",
    abstract = "Large language models (LLMs) have demonstrated superior performance to that of small language models (SLM) in information retrieval for various subtasks including dense retrieval, reranking, query expansion, and pseudo-document generation. However, the parameter sizes of LLMs are extremely large, making it expensive to operate LLMs stably for providing LLM-based retrieval services. Recently, retrieval-augmented language models have been widely employed to significantly reduce the parameter size by retrieving relevant knowledge from large-scale corpora and exploiting the resulting {``}in-context{''} knowledge as additional model input, thereby substantially reducing the burden of internalizing and retaining world knowledge in model parameters. Armed by the retrieval-augmented language models, we present a retrieval-augmented model specialization that distills the capability of LLMs to generate the chain-of-thoughts (CoT) for query expansion {--} that is, injects the LLM{'}s capability to generate CoT into a retrieval-augmented SLM {--} referred to as \textbf{RADCoT}. Experimental results on the MS-MARCO, TREC DL 19, 20 datasets show that RADCoT yields consistent improvements over distillation without retrieval, achieving comparable performance to that of the query expansion method using LLM-based CoTs. Our code is publicly available at \url{https://github.com/ZIZUN/RADCoT}.",
}
```

