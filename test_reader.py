
import torch
import transformers
import numpy as np
from pathlib import Path
import torch.distributed as dist
from torch.utils.data import DataLoader, SequentialSampler


import src.slurm
import src.util
from src.options import Options
import src.data
import src.evaluation
import src.model
from tqdm import tqdm

import json
import re
import collections
import string
import sys

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_tokens(s):
    if not s:
        return []
    return normalize_answer(s).split()


def compute_f1(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def evaluate(model, dataset, dataloader, tokenizer, opt):
    loss, curr_loss = 0.0, 0.0
    model.eval()
    if hasattr(model, "module"):
        model = model.module
    if opt.write_crossattention_scores:
        model.overwrite_forward_crossattention()
        model.reset_score_storage() 
    total = 0
    exactmatch = []
    f1= []
    if opt.write_results:
        write_path = Path(opt.checkpoint_dir) / opt.name / 'test_results'
        fw = open(write_path / ('%d.txt'%opt.global_rank), 'a')
    print(len(dataloader))
    with torch.no_grad():
        for i, batch in tqdm(enumerate(dataloader)):
            (idx, _, _, context_ids, context_mask) = batch

            if opt.write_crossattention_scores:
                model.reset_score_storage()

            outputs = model.generate(
                input_ids=context_ids.to("cuda"),
                attention_mask=context_mask.to("cuda"),
                max_length=50,
            )

            if opt.write_crossattention_scores:
                crossattention_scores = model.get_crossattention_scores(context_mask.to("cuda"))

            for k, o in enumerate(outputs):
                ans = tokenizer.decode(o, skip_special_tokens=True)
                example = dataset.data[idx[k]]
                if 'answers' in example:
                    score = src.evaluation.ems(ans, example['answers'])
                    f1_score = max(compute_f1(a, ans) for a in example['answers'])
                    
                    
                    exactmatch.append(score)
                    f1.append(f1_score)

                if opt.write_results:
                    fw.write(str(example['id']) + "\t" + ans + '\n')
                if opt.write_crossattention_scores:
                    for j in range(context_ids.size(1)):
                        example['ctxs'][j]['score'] = crossattention_scores[k, j].item()

                total += 1
            if (i + 1) % opt.eval_print_freq == 0:
                log = f'Process rank:{opt.global_rank}, {i+1} / {len(dataloader)}'
                if len(exactmatch) == 0:
                    log += '| no answer to compute scores'
                else:
                    log += f' | average = {np.mean(exactmatch):.3f}'
                logger.warning(log)


            
    logger.warning(f'Process rank:{opt.global_rank}, total {total} | average = {np.mean(exactmatch):.3f}')
    if opt.is_distributed:
        torch.distributed.barrier()
    score, total = src.util.weighted_average(np.mean(exactmatch), total, opt)
    f1score, _ = src.util.weighted_average(np.mean(f1), total, opt)
    
    return score, total, f1score


if __name__ == "__main__":
    options = Options()
    options.add_reader_options()
    options.add_eval_options()
    opt = options.parse()
    src.slurm.init_distributed_mode(opt)
    src.slurm.init_signal_handler()
    opt.train_batch_size = opt.per_gpu_batch_size * max(1, opt.world_size)

    dir_path = Path(opt.checkpoint_dir)/opt.name
    directory_exists = dir_path.exists()
    if opt.is_distributed:
        torch.distributed.barrier()
    dir_path.mkdir(parents=True, exist_ok=True)
    if opt.write_results:
        (dir_path / 'test_results').mkdir(parents=True, exist_ok=True)
    logger = src.util.init_logger(opt.is_main, opt.is_distributed, Path(opt.checkpoint_dir) / opt.name / 'run.log')
    if not directory_exists and opt.is_main:
        options.print_options(opt)


    tokenizer = transformers.AutoTokenizer.from_pretrained(opt.model_name, return_dict=False)


    collator_function = src.data.Collator(opt.text_maxlength, tokenizer)
    eval_examples = src.data.load_data(
        opt.eval_data, 
        global_rank=opt.global_rank, #use the global rank and world size attibutes to split the eval set on multiple gpus
        world_size=opt.world_size
    )
    eval_dataset = src.data.Dataset(
        eval_examples, 
        opt.n_context, 
    )

    eval_sampler = SequentialSampler(eval_dataset) 
    eval_dataloader = DataLoader(
        eval_dataset, 
        sampler=eval_sampler, 
        batch_size=opt.per_gpu_batch_size,
        num_workers=5, 
        collate_fn=collator_function
    )
    
    
    
    
    
    
    
    
    
    
    model_class = src.model.FiDT5
    model = model_class.from_pretrained(opt.model_path, ignore_mismatched_sizes=True)
    
    model = model.to(opt.device)
    
    

    logger.info("Start eval")
    exactmatch, total, f1_score = evaluate(model, eval_dataset, eval_dataloader, tokenizer, opt)

    logger.info(f'EM {100*exactmatch:.2f}, Total number of example {total}, F1 score - {f1_score}')

    if opt.write_results and opt.is_main:
        glob_path = Path(opt.checkpoint_dir) / opt.name / 'test_results'
        write_path = Path(opt.checkpoint_dir) / opt.name / 'final_output.txt'
        src.util.write_output(glob_path, write_path) 
    if opt.write_crossattention_scores:
        src.util.save_distributed_dataset(eval_dataset.data, opt)


















