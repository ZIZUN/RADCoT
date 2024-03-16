
from re import M
import time
import sys
import torch
import transformers
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler
from src.options import Options

import src.slurm
import src.util
import src.evaluation
import src.data
import src.model




def train(model, optimizer, scheduler, step, train_dataset, eval_dataset, opt, collator, best_dev_em, checkpoint_path):

    if opt.is_main:
        try:
            tb_logger = torch.utils.tensorboard.SummaryWriter(Path(opt.checkpoint_dir)/opt.name)
        except:
            tb_logger = None
            logger.warning('Tensorboard is not available.')

    torch.manual_seed(opt.global_rank + opt.seed) #different seed for different sampling depending on global_rank
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=opt.per_gpu_batch_size,
        drop_last=True,
        num_workers=3,
        collate_fn=collator
    )

    #################
    # model = model.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    # import torch.nn as nn
    # model = nn.DataParallel(model, device_ids=[0,1,2,3])
    #################

    loss, curr_loss = 0.0, 0.0
    epoch = 1
    model.train()
    while step < opt.total_steps:
        
        for i, batch in enumerate(train_dataloader):
            step += 1
            (idx, labels, _, context_ids, context_mask) = batch

            train_loss = model(
                input_ids=context_ids.cuda(),
                attention_mask=context_mask.cuda(),
                labels=labels.cuda(),
                # for new version transformers
                return_dict=True
            ).loss

            train_loss.mean().backward()

            if step % opt.accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), opt.clip)
                optimizer.step()
                scheduler.step()
                model.zero_grad()

            train_loss = src.util.average_main(train_loss, opt)
            curr_loss += train_loss.mean().item()

            if step % opt.eval_freq == 0:
                dev_em = evaluate(model, eval_dataset, tokenizer, collator, opt)
                model.train()
                if opt.is_main:
                    if dev_em > best_dev_em:
                        best_dev_em = dev_em
                        src.util.save(model, optimizer, scheduler, step, best_dev_em,
                                  opt, checkpoint_path, 'best_dev')
                    log = f"{step} / {opt.total_steps} |"
                    log += f"train: {curr_loss/opt.eval_freq:.3f} |"
                    log += f"evaluation: {100*dev_em:.2f}Rouge-L-F |"
                    log += f"lr: {scheduler.get_last_lr()[0]:.5f}"
                    logger.info(log)    
                    if tb_logger is not None:
                        tb_logger.add_scalar("Evaluation", dev_em, step)
                        tb_logger.add_scalar("Training", curr_loss / (opt.eval_freq), step)
                    curr_loss = 0.

            if opt.is_main and step % opt.save_freq == 0:
                src.util.save(model, optimizer, scheduler, step, best_dev_em,
                          opt, checkpoint_path, f"step-{step}")
                
            
            if step > opt.total_steps:
                logger.info(f'{epoch} epoch, {step % opt.accumulation_steps} steps Complete !! ')    
                break
        epoch += 1
        logger.info(f'{epoch} epoch, {step % opt.accumulation_steps} steps Complete !! ')    

def evaluate(model, dataset, tokenizer, collator, opt):
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset,
        sampler=sampler,
        batch_size=opt.per_gpu_batch_size,
        drop_last=False,
        num_workers=10,
        collate_fn=collator
    )
    model.eval()
    total = 0
    exactmatch = []
    model = model.module if hasattr(model, "module") else model
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            (idx, _, _, context_ids, context_mask) = batch

            outputs = model.generate(
                input_ids=context_ids.cuda(),
                attention_mask=context_mask.cuda(),
                max_length=150,
            )
            # print(outputs)

            for k, o in enumerate(outputs):
                ans = tokenizer.decode(o, skip_special_tokens=True)
                gold = dataset.get_example(idx[k])['answers'][0]
                ####
                # if k == 0 and i ==0:
                #     print('---ans---')
                #     print(ans)
                #     print('---gold---')
                #     print(gold)
                #     print('-------------------------')
                # score = src.evaluation.ems(ans, gold)
                try:
                    score = src.evaluation.get_rouge_score(ans, gold)
                except:
                    score = 0
                total += 1
                exactmatch.append(score)

    exactmatch, total = src.util.weighted_average(np.mean(exactmatch), total, opt)
    return exactmatch

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')    
    
    options = Options()
    options.add_reader_options()
    options.add_optim_options()
    opt = options.parse()
    #opt = options.get_options(use_reader=True, use_optim=True)

    torch.manual_seed(opt.seed)
    src.slurm.init_distributed_mode(opt)
    src.slurm.init_signal_handler()

    checkpoint_path = Path(opt.checkpoint_dir)/opt.name
    checkpoint_exists = checkpoint_path.exists()
    if opt.is_distributed:
        torch.distributed.barrier()
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    #if not checkpoint_exists and opt.is_main:
    #    options.print_options(opt)
    #checkpoint_path, checkpoint_exists = util.get_checkpoint_path(opt)

    logger = src.util.init_logger(
        opt.is_main,
        opt.is_distributed,
        checkpoint_path / 'run.log'
    )

    model_name = 'google/flan-t5-' + opt.model_size
    model_class = src.model.FiDT5

    #load data
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    
    collator = src.data.Collator(opt.text_maxlength, tokenizer, answer_maxlength=opt.answer_maxlength)

    # use golbal rank and world size to split the eval set on multiple gpus
    train_examples = src.data.load_data(
        opt.train_data, 
        global_rank=opt.global_rank, 
        world_size=opt.world_size,
    )
    train_dataset = src.data.Dataset(train_examples, opt.n_context)
    # use golbal rank and world size to split the eval set on multiple gpus
    eval_examples = src.data.load_data(
        opt.eval_data,
        global_rank=opt.global_rank,
        world_size=opt.world_size,
    )
    eval_dataset = src.data.Dataset(eval_examples, opt.n_context)


    t5 = transformers.T5ForConditionalGeneration.from_pretrained(model_name)
    # t5.resize_token_embeddings(len(tokenizer)) ####
    
    if opt.model_path != "none":
        t5.load_state_dict(torch.load(opt.model_path, map_location=torch.device('cpu')))
        print('Pretrained model load complete!!')
    
    model = src.model.FiDT5(t5.config)
    model.load_t5(t5.state_dict())
    
    # model.from_pretrained(opt.model_path)
    
    model = model.to(opt.local_rank)
    optimizer, scheduler = src.util.set_optim(opt, model)
    step, best_dev_em = 0, 0.0

    model.set_checkpoint(opt.use_checkpoint)

    if opt.is_distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[opt.local_rank],
            output_device=opt.local_rank,
            find_unused_parameters=False,
        )

    logger.info("Start training")
    train(
        model,
        optimizer,
        scheduler,
        step,
        train_dataset,
        eval_dataset,
        opt,
        collator,
        best_dev_em,
        checkpoint_path
    )
