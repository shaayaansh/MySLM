import os
import torch
import numpy as np
from tqdm import tqdm
from torch.nn import DataParallel
from utils import tokenize_train_data
from bpe_tokenizer import Tokenizer
from transformers import GPT2Tokenizer
from torch.utils.data import DataLoader
from models import TransformerLM

    
def main():
    TRAIN_PATH = "data/TinyStoriesV2-GPT4-train.txt"
    tokens_path = "train_tokens.bin"

    batch_size = 128
    context_length = 256

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(tokens_path):
        tokenize_train_data(TRAIN_PATH)

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    eos_token_id = tokenizer.eos_token_id

    model = TransformerLM(
        vocab_size=tokenizer.vocab_size,
        context_length=context_length,
        d_model=512,
        num_layers=4,
        num_heads=16,
        d_ff=1344,
        rope_theta=10000,
    )
    model = DataParallel(model, device_ids=[0, 1]).to(device)
    
    # load all tokens
    tokens = np.memmap(tokens_path, dtype=np.uint16, mode='r')

    for idx in tqdm(range(len(tokens) // batch_size), desc="training ..."):
        batch = torch.tensor(tokens[idx*context_length*batch_size: (idx+1)*context_length*batch_size], dtype=torch.long)
        batch_reshaped = batch.reshape(batch_size, context_length).to(device)
        
        x = batch_reshaped[:, :-1]
        target = batch_reshaped[:, 1:]

        outputs = model(x)

        

if __name__ == "__main__":
    main()






    







        

            
