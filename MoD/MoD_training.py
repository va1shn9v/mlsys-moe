#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os

import torch
from torch.utils.data import DataLoader

from datasets import load_dataset

from fairscale.nn.model_parallel.initialize import (
    initialize_model_parallel,
    model_parallel_is_initialized,
)
from llama.tokenizer import Tokenizer

from mixture_of_depths.routing_transformer import ModelArgs, MoDTransformer
from mixture_of_depths.train import MoDLlamaTrainer


# In[2]:


if not torch.distributed.is_initialized():
    # torch.distributed.init_process_group("nccl")
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'
    torch.distributed.init_process_group(backend='gloo', rank=0, world_size=1)
if not model_parallel_is_initialized():
    model_parallel_size = int(os.environ.get("WORLD_SIZE", 1))
    initialize_model_parallel(model_parallel_size)

local_rank = int(os.environ.get("LOCAL_RANK", 0))
torch.cuda.set_device(local_rank)
torch.manual_seed(42)
torch.set_default_tensor_type(torch.cuda.FloatTensor)


# ## Load Data

# In[3]:


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
    batch_size=4,
    collate_fn=collate_fn,
)




def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ### With MoD and Auxiliary Loss

# In[8]:


# model_params = ModelArgs(
#     dim=512,
#     n_layers=12,
#     n_heads=8,
#     vocab_size=tokenizer.n_words,
#     routing=True,
#     aux_loss=True
# )
# model = MoDTransformer(model_params)




# print("Model Param Count : {}".format(count_parameters(model)))



# trainer = MoDLlamaTrainer(
#     params=model_params,
#     model=model,
#     tokenizer=tokenizer,
#     dataloader=dataloader
# )


# In[11]:


# get_ipython().run_cell_magic('time', '', 'trainer.train(\n    epochs=5,\n    use_aux_loss=model_params.aux_loss,\n    use_aux_predictor=model.params.aux_routing\n)\n')

# trainer.train(epochs=7,use_aux_loss=model_params.aux_loss,use_aux_predictor=model.params.aux_routing)
# ### With MoD and Auxiliary Router

# In[9]:


model_params = ModelArgs(
    dim=1024,
    n_layers=12,
    n_heads=8,
    vocab_size=tokenizer.n_words,
    routing=True,
    aux_routing=True
)
model = MoDTransformer(model_params)
print("Model Params in  Aux predictor model : {}".format(count_parameters(model)))


# In[10]:


trainer = MoDLlamaTrainer(
    params=model_params,
    model=model,
    tokenizer=tokenizer,
    dataloader=dataloader
)


trainer.train(epochs=5,model_dir="./models/MoDLlama_predictor/",log_path="./logs/MoDLlama_predictor_log.txt",use_aux_loss=model_params.aux_loss,use_aux_predictor=model.params.aux_routing)



# model_params = ModelArgs(
#     dim=512,
#     n_layers=6,
#     n_heads=8,
#     vocab_size=tokenizer.n_words,
#     routing=False,
# )
# model = MoDTransformer(model_params)
# count_parameters(model)



# trainer = MoDLlamaTrainer(
#     params=model_params,
#     model=model,
#     tokenizer=tokenizer,
#     dataloader=dataloader
# )

# trainer.train(epochs=5, model_dir="./models/BaselineLlama/",log_path="./logs/BaselineLlama_log.txt")






