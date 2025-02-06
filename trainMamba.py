import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
sys.path.append('..')

import shutil

from time import perf_counter

import torch
import torch.nn.functional as F
import torch.nn as nn
from mamba_lm import from_pretrained
from mamba_lm import MambaLM, MambaLMConfig

import datasets

import numpy as np
import random

# Automated device selection based on available backends
device = torch.device("cpu")

print(f"> Using {device} device")

def listdir_nohidden(path):
    files = []
    for f in os.listdir(path):
        if not f.startswith('.'):
            files.append(f"{path}/{f}")
    return files

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def load_checkpoint(filepath, model, scheduler, optimizer):
    print(f"> Loading model from: {filepath}")
    try:
        loaded_checkpoint = torch.load(filepath, map_location=device)

        loaded_epoch = loaded_checkpoint['epoch']
        loaded_model = model
        loaded_scheduler = scheduler
        loaded_optimizer = optimizer

        loaded_model.load_state_dict(loaded_checkpoint['model_state'])
        if scheduler is not None:
            loaded_scheduler.load_state_dict(loaded_checkpoint['scheduler_state'])
        if optimizer is not None:
            loaded_optimizer.load_state_dict(loaded_checkpoint['optimizer_state'])
        
        print("> Loaded model")
        return True, loaded_epoch, loaded_model, loaded_scheduler, loaded_optimizer
    except Exception as e:
        print("> Cannot load model")
        return False, 0, model, scheduler, optimizer

def train(pretrained=False):
    epochs = 10
    batch_size = 32 
    seq_length = 128
    learning_rate = 10e-3
    model_path = f'saves/model.pth'
    backup_path = f"saves/model-b.pth"

    # Load dataset
    dataset = datasets.load_dataset('text', data_files={'train': ['aranyakanda_english.txt']})

    # Initialize the BPE tokenizer
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors

    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.BpeTrainer(vocab_size=5000, min_frequency=2)
    tokenizer.train_from_iterator(dataset['train']['text'], trainer=trainer)
    
    # Save the tokenizer for future use
    tokenizer.save("bpe_tokenizer.json")
    
    # Load tokenizer again
    tokenizer = Tokenizer.from_file("bpe_tokenizer.json")
    tokenizer.enable_padding(pad_id=0, pad_token='<pad>', pad_width=1)
    tokenizer.enable_truncation(max_length=seq_length)

    # Tokenize data
    def tokenize_data(example):
        encoding = tokenizer.encode(example['text'])
        return {'tokens': encoding.tokens}

    tokenized_dataset = dataset.map(tokenize_data, remove_columns=['text'])

    vocab = tokenizer.get_vocab()
    print(f"Vocab size: {len(vocab)}")

    if pretrained:
        model = from_pretrained('state-spaces/mamba-130m').to(device)
    else:
        config = MambaLMConfig(d_model=16, n_layers=4, vocab_size=len(vocab))
        model = MambaLM(config).to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, mode='min', factor=0.1, patience=2)

    _, epoch, model, scheduler, optim = load_checkpoint(model_path, model, scheduler, optim)

    def get_data(dataset, vocab, batch_size):
        data = []                                   
        for example in dataset:
            if example['tokens']:
                tokens = [vocab[token] for token in example['tokens']]
                data.extend(tokens)
        
        data = torch.LongTensor(data)              
        num_batches = data.shape[0] // batch_size 
        data = data[:num_batches * batch_size]                       
        data = data.view(batch_size, num_batches)
        return data     

    def get_batch(data, seq_len, idx):
        src = data[:, idx:idx+seq_len]
        target = data[:, idx+1:idx+seq_len+1]
        return src, target


    train_data = get_data(tokenized_dataset['train'], vocab, batch_size)
    print(f"Train data length before: {train_data.shape[-1]}")

    t0_start = perf_counter()
    for z in range(epoch, epochs):
        idx = 0
        avg_loss = 0
        print(f"\n> Epoch {z+1}/{epochs}")

        t2_start = perf_counter()
        for i in range(train_data.shape[-1]):   
            model.train()
            t1_start = perf_counter()

            input, output = get_batch(train_data, seq_length, idx)
            output = output.reshape(-1)
            input = input.to(device)
            output = output.to(device)

            logits = model(input)

            if (logits.view(-1, logits.size(-1)).shape[0] != output.view(-1).shape[0]):
                print("skip")
            else:
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), output)
                avg_loss += loss.item()

                optim.zero_grad()
                loss.backward()
                optim.step()

                t1_stop = perf_counter()

                if i%10==0:
                    print(f"\r> Batch: {idx}/{train_data.shape[-1]-seq_length} loss: {avg_loss/(i+1):.5f} time: {t1_stop-t1_start:.2f} sec ", end="")

                    checkpoint = {
                        'epoch': z,
                        'model_state': model.state_dict(),
                        'optimizer_state': optim.state_dict(),
                        'scheduler_state': scheduler.state_dict(),
                    }
                    if backup_path is not None and os.path.isfile(model_path):
                        shutil.copyfile(model_path, backup_path)
                    torch.save(checkpoint, model_path)

            idx += 1
            if idx >= train_data.shape[-1] - seq_length:
                idx = 0
                break

        t2_stop = perf_counter()
        print(f"\n> Epoch time: {t2_stop - t2_start:.3f} seconds")
        scheduler.step(avg_loss/(i+1))

    t0_stop = perf_counter()
    print(f"\n> Finished training in: {t0_stop-t0_start} seconds")

    print("> Generating answer: ")
    # Tokenize the prompt
    encoded_input = tokenizer.encode("Description Of Rama")
    input_ids = torch.tensor(encoded_input.ids).unsqueeze(0).to(device)  # Add batch dimension

    # Generate output using the precomputed input_ids
    output = model.generate(tokenizer=tokenizer, input_ids=input_ids, num_tokens=50, temperature=1.0, top_k=None)

    print(f"Answer: {output}")


def my_gen(pretrained=False):
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors
    tokenizer = Tokenizer.from_file("bpe_tokenizer.json")
    tokenizer.enable_padding(pad_id=0, pad_token='<pad>', pad_width=1)
    tokenizer.enable_truncation(max_length=32)

    if pretrained:
        model = from_pretrained('state-spaces/mamba-130m').to(device)
    else:
        config = MambaLMConfig(d_model=16, n_layers=4, vocab_size=len(tokenizer.get_vocab()))
        model = MambaLM(config).to(device)

    isLoaded, _, model, *_ = load_checkpoint(f'saves/model.pth', model, None, None)
    if (not isLoaded):
        return

    encoded_input = tokenizer.encode("Kaikeyi is one of the three queens")
    input_ids = torch.tensor(encoded_input.ids).unsqueeze(0).to(device)  # Add batch dimension

    # Generate output using the precomputed input_ids
    output = model.generate(tokenizer=tokenizer, input_ids=input_ids, num_tokens=10, temperature=1.0, top_k=None)


    print(f"Answer: {output}")


def prepare_folders():
    try:
        os.makedirs("./saves/")
    except:
        pass

if __name__ == "__main__":
    seed_everything(555)
    prepare_folders()

    # train(pretrained=False)
    my_gen(pretrained=False)
