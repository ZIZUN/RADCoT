from pyserini.search import LuceneSearcher, get_topics, get_qrels
import tempfile
from src.trec_eval import run_retriever, EvalFunction
import json

import argparse


parser = argparse.ArgumentParser()
parser.add_argument("-e", "--eval_dataset", required=True, type=str, help="eval dataset")
parser.add_argument("-q", "--query_augment_file_path", required=True, type=str, help="query_augment_file_path")

args = parser.parse_args()
args.query_augment_file_path


# parameter setup
if args.eval_dataset == 'trec-dl-19':
    data_num = 19
elif args.eval_dataset == 'trec-dl-20':
    data_num = 20
elif args.eval_dataset == 'ms-marco-dev':
    data_num = 21

# load query augmentation file
if args.query_augment_file_path is not None:
    query_augment_results = []
    with open(args.query_augment_file_path, "r") as f:
        for line in f:
            per_result = line.strip().split('\t')
            try:
                query_augment_results.append(per_result[1])
            except:
                query_augment_results.append('')



# Retrieve passages using pyserini BM25.
searcher = LuceneSearcher.from_prebuilt_index('msmarco-v1-passage')
if data_num == 19:
    topics = get_topics('dl19-passage')
elif data_num == 20:
    topics = get_topics('dl20')    
else:
    topics = get_topics('msmarco-passage-dev-subset')    
    
    

####  for Query expansion

k = 5
for i, key in enumerate(list(topics.keys())):    
    if args.query_augment_file_path is not None:
        aug_num = len(query_augment_results[i]) // len(topics[key]['title'])
        
        if aug_num < k:
            aug_num = k
        topics[key]['title'] = (topics[key]['title']+ ' ') * aug_num + ' ' + query_augment_results[i]
    else:
        topics[key]['title'] = topics[key]['title']
    

qrels = get_qrels(f'dl{data_num}-passage')
if data_num == 21:
    rank_results = run_retriever(topics, searcher, qrels, k=1000)
else:
    rank_results = run_retriever(topics, searcher, qrels, k=100)


# # # #######################################################
# # # save retrieval result
# ret_results = []
# for result in rank_results:
#     question = result['query']
#     answers = ['']
#     ctxs = []
    
    
    
#     for hit in result['hits']:
#         # ctxs.append({'id': hit['rank'], 'title': '', 'text': f"[{hit['rank']}] " + hit['content']})
#         ctxs.append({'id': hit['rank'], 'title': '', 'text': f"Passage: " + hit['content']})
    
#     ret_results.append({'question': question, 'answers': answers, 'ctxs': ctxs})

# with open(f'', 'w') as outfile:
#     json.dump(ret_results, outfile, indent=4)

# exit()
# # # #######################################################


def write_eval_file(rank_results, file):
    with open(file, 'w') as f:
        for i in range(len(rank_results)):
            rank = 1
            hits = rank_results[i]['hits']
            for hit in hits:
                f.write(f"{hit['qid']} Q0 {hit['docid']} {rank} {hit['score']} rank\n")
                rank += 1
    return True




# Evaluate nDCG@(num)
temp_file = tempfile.NamedTemporaryFile(delete=False).name
write_eval_file(rank_results, temp_file)

if data_num == 19 or data_num==20:
    
    EvalFunction.eval(['-c', '-m', f'ndcg_cut.1', f'dl{data_num}-passage', temp_file])
    EvalFunction.eval(['-c', '-m', f'ndcg_cut.5', f'dl{data_num}-passage', temp_file])
    EvalFunction.eval(['-c', '-m', f'ndcg_cut.10', f'dl{data_num}-passage', temp_file])   # dl19-passage  dl20-passage
    
    # ###
    # EvalFunction.eval(['-c', '-m', f'recall.10', f'dl{data_num}-passage', temp_file])
    # EvalFunction.eval(['-c', '-m', f'recall.50', f'dl{data_num}-passage', temp_file])
    # EvalFunction.eval(['-c', '-m', f'recall.100', f'dl{data_num}-passage', temp_file])
    # EvalFunction.eval(['-c', '-M', '1', '-m', f'recip_rank',  f'dl{data_num}-passage', temp_file])
    # EvalFunction.eval(['-c', '-M', '5', '-m', f'recip_rank',  f'dl{data_num}-passage', temp_file])
    # EvalFunction.eval(['-c', '-M', '10', '-m', f'recip_rank',  f'dl{data_num}-passage', temp_file])
else:
    EvalFunction.eval(['-c', '-M', '10', '-m', f'recip_rank', f'msmarco-passage-dev-subset', temp_file])
    EvalFunction.eval(['-c', '-m', f'recall.50', f'msmarco-passage-dev-subset', temp_file])
    EvalFunction.eval(['-c', '-m', f'recall.1000', f'msmarco-passage-dev-subset', temp_file])   
    