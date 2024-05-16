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
import argparse
from functools import partial
import logging


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def collate_fn(batch,tokenizer):
    bsz = len(batch)    
    tokenized_texts = [tokenizer.encode(x['text'], bos=True, eos=True) for x in batch]
    max_text_len = max(len(t) for t in tokenized_texts)
    pad_id = tokenizer.eos_id
    tokens = torch.full((bsz, min(2048, max_text_len)), pad_id, dtype=torch.long)
    for k, t in enumerate(tokenized_texts):
        tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long)[:2048]
    
    return tokens[:,:-1], tokens[:,1:]


tokenizer = Tokenizer("tokenizer.model")
collate_with_tokenizer = partial(collate_fn, tokenizer=tokenizer)   

def train(args):
    
    if not torch.distributed.is_initialized():
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

 
    dataset = load_dataset("roneneldan/TinyStories", split="train")
    
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        collate_fn=collate_with_tokenizer,
    )

    model_params = ModelArgs(
        dim=args.dim,
        n_layers=args.n_layers,
        n_heads=8,
        vocab_size=tokenizer.n_words,
        routing= args.routing,
        aux_loss= args.aux_loss,
        aux_routing=args.aux_routing,
        router_skip_blocks= args.router_skip_blocks
    )

    model = MoDTransformer(model_params)
    print(f'model arch dim:{model_params.dim}, n_layers:{model_params.n_layers}, n_heads:{model_params.n_heads}')
    print("count", count_parameters(model))


    trainer = MoDLlamaTrainer(
        params=model_params,
        model=model,
        tokenizer=tokenizer,
        dataloader=dataloader
    )

    trainer.train(
        epochs=args.epochs,
        model_dir=args.model_dir,
        use_aux_loss=args.aux_loss,
        use_aux_predictor=args.aux_routing, 
        save_routing_path = args.routing_decision_path_dir,
        use_wandb=args.use_wandb,
        aux_epochs=args.aux_epochs
    )

def main():
    
    parser = argparse.ArgumentParser(description='MoD Training')
    parser.add_argument('--aux_epochs', type=int, default=5, help='')
    parser.add_argument('--use_wandb', type=bool, default=True, help='')
    parser.add_argument('--router_skip_blocks', type=int, default=2, help='No of blocks to skip (default: 2)')
    parser.add_argument('--aux_loss', type=bool, default=False, help='Enable auxiliary loss (default: False)')
    parser.add_argument('--aux_routing', type=bool, default=False, help='Enable auxiliary loss (default: False)')
    parser.add_argument('--routing', type=str, default='True', help='Enable routing (default: False)')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs (default: 10)')
    parser.add_argument('--model_dir', type=str, default='/scratch/sxp8182/MoD_simran_2/models/default', help='Number of epochs (default: 10)')
    parser.add_argument('--log_path', type=str, default='', help='Number of epochs (default: 10)')
    parser.add_argument('--log_path_debug', type=str, default='output.log', help='Number of epochs (default: 10)')
    parser.add_argument('--routing_decision_path_dir', type=str, default='/scratch/sdm8499/MOD/mlsys-moe/MoD/mixture_of_depths/plots/', help='Number of epochs (default: 10)')
    parser.add_argument('--dim', type=int, default=512, help='Hidden dim (default: 512)')
    parser.add_argument('--n_layers', type=int, default=6, help='Number of layers (default: 6)')
    
    args = parser.parse_args()
   
    args.routing = args.routing.lower() in ['true', '1', 't', 'y', 'yes']
    logging.basicConfig(filename=args.log_path_debug, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',filemode = 'w')
    logger = logging.getLogger(__name__)
    
    logger.info(args)
    train(args)

if __name__ == '__main__':
    main()
    
