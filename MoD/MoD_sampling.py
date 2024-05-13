#!/usr/bin/env python
# coding: utf-8




from mixture_of_depths.generation import MoDLlama
from torch.profiler import profile, record_function, ProfilerActivity




# generator = MoDLlama.build(
#     ckpt_dir="./models/BaselineLlama/",
#     tokenizer_path="../llama/tokenizer.model",
#     max_seq_len=2048,
#     max_batch_size=1,
# )
# generator.text_completion(
#     [
#         "On a sunny day",
#     ]
# )


# Auxiliary Loss



generator = MoDLlama.build(
    ckpt_dir="./models/MoDLlama/",
    tokenizer_path="tokenizer.model",
    max_seq_len=2048,
    max_batch_size=4,
    model_parallel_size=1
)
# with profile(activities=[
#         ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
#     with record_function("model_inference"):
#         # model(inputs)
# gen = generator.text_completion(
# [
# "John and Sarah were playing together in their backyard when they found a piece of metal. It was shiny and reflective and they couldn't wait to show their parents. John asked Sarah","On a sunny Day","Ona Rainy Day","On a stormy Day I went to school"
# ]
# )
# print(gen)
gen = generator.text_completion(
[
"John and Sarah were playing together in their backyard when they found a piece of metal. It was shiny and reflective and they couldn't wait to show their parents. John asked Sarah",
]
)
print(gen)
# print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))




# Auxiliary Router (currently doesn't seem that great)

# In[2]:


# generator = MoDLlama.build(
#     ckpt_dir="./models/MoDLlama_predictor/",
#     tokenizer_path="tokenizer.model",
#     max_seq_len=2048,
#     max_batch_size=1,
# )
# print(generator.text_completion(
#     [
#         "On a sunny day",
#     ]
# ))





