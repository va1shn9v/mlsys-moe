from mixture_of_depths.generation_v3 import MoDLlama
from torch.profiler import profile, record_function, ProfilerActivity
import os
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from llama.tokenizer import Tokenizer
from functools import partial
import random
import logging 

logging.basicConfig(
    filename='decoding_steps.log', 
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='a'
)
logger = logging.getLogger('logger')

generator = MoDLlama.build(
    ckpt_dir="/scratch/sdm8499/sushi/MoD_simran_2/model_aux_routing_10_RR_143",
    tokenizer_path="tokenizer.model",
    max_seq_len=2048,
    max_batch_size=64,
    model_parallel_size=1
)

def collate_fn(batch):
    return [item['text'] for item in batch]



tokenizer = Tokenizer("tokenizer.model")

def sample_random_prompts_from_train_data():
    
    dataset = load_dataset("roneneldan/TinyStories", split="train")

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        collate_fn=collate_fn,
    )
    
    collected_texts = []
    for batch in dataloader:
        collected_texts.extend(batch)
        if len(collected_texts) >= 200:
            break
        
    print(len(collected_texts), collected_texts[0])
    
    minl = 100000
    
    for i in range(0,len(collected_texts)):
        if minl > len(collected_texts[i]):
            minl = len(collected_texts[i])
            
    print(minl)
    
    return collected_texts, minl

def get_routing_distribution(seed=None):
    if seed is not None:
        random.seed(seed)
    
    collected_text, minl = sample_random_prompts_from_train_data()
    split_idx = 100
    collected_text = [x[:split_idx] for x in collected_text]

    run = 1
    while run <= 50:
        bsz = random.randint(1, 64) 
        
        print("Batchsize",bsz)
        
        input_list = random.sample(collected_text, bsz)
        
        gen = generator.text_completion(input_list)
        
        run += 1

seed = 42
#get_routing_distribution(seed)
  
def get_execution_time(seed=None):
    if seed is not None:
        random.seed(seed)
    
    collected_text, minl = sample_random_prompts_from_train_data()
    split_idx = 100
    collected_text = [x[:split_idx] for x in collected_text]

    for bsz in range(1,64):
        
        input_list = random.sample(collected_text, bsz)
        
        gen = generator.text_completion(input_list)
    
# get_execution_time(seed)  
    
    
def get_out_sequence_distribution(seed=None):
    if seed is not None:
        random.seed(seed)
    
    collected_text, minl = sample_random_prompts_from_train_data()
    split_idx = 100
    collected_text = [x[:split_idx] for x in collected_text]

    run = 1
    out_sequence_dict = {}
    
    for bsz in range(1,64):
        logger.info(f'bsz{bsz}')
        input_list = random.sample(collected_text, bsz)
        
        gen = generator.text_completion(input_list)
        
        len1 = []
       
        for i in range(bsz):
            len1.append(len(gen[i]['generation'].split()))
            
        out_sequence_dict[bsz] = len1
        
    with open("out_sequence_dict.txt", "w") as file:
        for key, value in out_sequence_dict.items():
            file.write(f"{key}: {value}\n")

get_out_sequence_distribution(seed)