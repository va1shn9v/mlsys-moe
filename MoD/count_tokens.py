from datasets import load_dataset
from transformers import AutoTokenizer
from llama.tokenizer import Tokenizer


# Step 1: Load the dataset
dataset = load_dataset("roneneldan/TinyStories")
tokenizer = Tokenizer("tokenizer.model")

# Step 2: Load a tokenizer
# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

# Step 3: Define a function to tokenize and count tokens
def tokenize_and_count(text):
    tokens = tokenizer.encode(text,bos=False,eos=False)
    # print(tokens)
    return {"text":len(tokens)}

# Step 4: Apply the function to the dataset and sum all tokens
total_tokens = sum(dataset.map(lambda x: tokenize_and_count(x['text']), batched=False))

print("Total number of tokens:", total_tokens)