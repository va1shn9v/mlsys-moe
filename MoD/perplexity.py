#!/usr/bin/env python
# coding: utf-8




from mixture_of_depths.generation import MoDLlama
from torch.profiler import profile, record_function, ProfilerActivity
from torch.utils.data import DataLoader

from datasets import load_dataset
from llama.tokenizer import Tokenizer
import torch
import torch.cuda as cuda

cuda.reset_peak_memory_stats(device='cuda')



dataset = load_dataset("roneneldan/TinyStories", split="train")

tokenizer = Tokenizer("tokenizer.model")
def collate_fn(batch):
    bsz = len(batch)
    tokenized_texts = [tokenizer.encode(x['text'], bos=True, eos=True) for x in batch]
    max_text_len = max(len(t) for t in tokenized_texts)

    pad_id = tokenizer.eos_id
    tokens = torch.full((bsz, min(2048, max_text_len)), pad_id, dtype=torch.long)
    for k, t in enumerate(tokenized_texts):
        tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long)[:2048]
    
    return tokens[:,:-1], tokens[:,1:]

dataloader = DataLoader(
    dataset,
    batch_size=1,
    collate_fn=collate_fn,
)

generator = MoDLlama.build(
    ckpt_dir="./models/BaselineLlama/",
    tokenizer_path="tokenizer.model",
    max_seq_len=2048,
    max_batch_size=1,
)
it  = iter(dataloader)
avg_perp = 0
num_sample = 5000
for i in range(num_sample):
    inputs,targets = next(it)
    perplexity = generator.cal_perplexity(inputs,targets)
    current_memory = cuda.memory_allocated()
    peak_memory = cuda.max_memory_allocated()

    print(f"Current memory: {current_memory}, Peak memory: {peak_memory}")

    print(perplexity)
    avg_perp+= perplexity
print("Avg Perplexity : {}".format(avg_perp/num_sample))



