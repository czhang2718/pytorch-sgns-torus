# -*- coding: utf-8 -*-

import os
import pickle
import random
import argparse
import torch as t
import numpy as np

from tqdm import tqdm
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from model import Word2Vec, SGNS

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Install with: pip install wandb")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='sgns', help="model name")
    parser.add_argument('--data_dir', type=str, default='./data/', help="data directory path")
    parser.add_argument('--save_dir', type=str, default='./pts/', help="model directory path")
    parser.add_argument('--e_dim', type=int, default=300, help="embedding dimension")
    parser.add_argument('--n_negs', type=int, default=20, help="number of negative samples")
    parser.add_argument('--epoch', type=int, default=100, help="number of epochs")
    parser.add_argument('--mb', type=int, default=4096, help="mini-batch size")
    parser.add_argument('--ss_t', type=float, default=1e-5, help="subsample threshold")
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
    parser.add_argument('--conti', action='store_true', help="continue learning")
    parser.add_argument('--weights', action='store_true', help="use weights for negative sampling")
    parser.add_argument('--cuda', action='store_true', help="use CUDA")
    parser.add_argument('--torus', action='store_true', default=False, help="train toric embeddings")
    parser.add_argument('--wandb', action='store_true', help="log to Weights & Biases")
    parser.add_argument('--wandb_name', type=str, default='pytorch-sgns-torus', help="wandb run name")
    parser.add_argument('--wandb_project', type=str, default='pytorch-sgns-torus', help="wandb project name")
    parser.add_argument('--wandb_entity', type=str, default=None, help="wandb entity/team name")
    parser.add_argument('--wandb_log_interval', type=int, default=1, help="log to wandb every N batches")
    return parser.parse_args()


class PermutedSubsampledCorpus(Dataset):

    def __init__(self, datapath, ws=None):
        data = pickle.load(open(datapath, 'rb'))
        if ws is not None:
            self.data = []
            for iword, owords in data:
                if random.random() > ws[iword]:
                    self.data.append((iword, owords))
        else:
            self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        iword, owords = self.data[idx]
        return iword, np.array(owords)


def train(args):
    idx2word = pickle.load(open(os.path.join(args.data_dir, 'idx2word.dat'), 'rb'))
    wc = pickle.load(open(os.path.join(args.data_dir, 'wc.dat'), 'rb'))
    wf = np.array([wc[word] for word in idx2word])
    wf = wf / wf.sum()
    ws = 1 - np.sqrt(args.ss_t / wf)
    ws = np.clip(ws, 0, 1)
    vocab_size = len(idx2word)
    weights = wf if args.weights else None
    if not os.path.isdir(args.save_dir):
        os.mkdir(args.save_dir)
    model = Word2Vec(vocab_size=vocab_size, embedding_size=args.e_dim, torus=args.torus)
    modelpath = os.path.join(args.save_dir, '{}.pt'.format(args.name))
    sgns = SGNS(embedding=model, vocab_size=vocab_size, n_negs=args.n_negs, weights=weights, torus=args.torus)
    if os.path.isfile(modelpath) and args.conti:
        sgns.load_state_dict(t.load(modelpath))
    if args.cuda:
        sgns = sgns.cuda()
    optim = Adam(sgns.parameters(), lr=args.lr)
    optimpath = os.path.join(args.save_dir, '{}.optim.pt'.format(args.name))
    if os.path.isfile(optimpath) and args.conti:
        optim.load_state_dict(t.load(optimpath))
    
    # Initialize wandb
    if args.wandb and WANDB_AVAILABLE:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_name,
            config={
                'vocab_size': vocab_size,
                'embedding_dim': args.e_dim,
                'n_negs': args.n_negs,
                'epochs': args.epoch,
                'batch_size': args.mb,
                'learning_rate': args.lr,
                'subsample_threshold': args.ss_t,
                'use_weights': args.weights,
                'torus': args.torus,
                'cuda': args.cuda,
            }
        )
    
    best_loss = float('inf')
    tokens_seen = 0
    
    for epoch in range(1, args.epoch + 1):
        dataset = PermutedSubsampledCorpus(os.path.join(args.data_dir, 'train.dat'))
        dataloader = DataLoader(dataset, batch_size=args.mb, shuffle=True)
        
        epoch_losses = []
        pbar = tqdm(dataloader)
        pbar.set_description("[Epoch {}]".format(epoch))
        
        for batch_idx, (iword, owords) in enumerate(pbar, 1):
            loss = sgns(iword, owords)
            optim.zero_grad()
            loss.backward()
            optim.step()
            
            loss_val = loss.item()
            epoch_losses.append(loss_val)
            
            # Count tokens: batch_size * context_size
            batch_size = iword.size(0)
            context_size = owords.size(1)
            tokens_seen += batch_size * context_size
            
            # Check for NaN or Inf (bug detection)
            if not np.isfinite(loss_val):
                print(f"\nWARNING: Non-finite loss detected at epoch {epoch}, batch {batch_idx}: {loss_val}")
                if args.wandb and WANDB_AVAILABLE:
                    wandb.alert(title="NaN/Inf Loss Detected", text=f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss_val}")
                break
            
            pbar.set_postfix(loss=loss_val)
            
            # Log to wandb at specified intervals
            if args.wandb and WANDB_AVAILABLE and (batch_idx % args.wandb_log_interval == 0):
                wandb.log({'loss': loss_val, 'epoch': epoch}, step=tokens_seen)
        
        # Log at end of epoch
        epoch_avg_loss = np.mean(epoch_losses)
        if epoch_avg_loss < best_loss:
            best_loss = epoch_avg_loss
        print(f"[Epoch {epoch}/{args.epoch}] Average Loss: {epoch_avg_loss:.4f}, Best Loss: {best_loss:.4f}")
        
        # Log epoch average to wandb
        if args.wandb and WANDB_AVAILABLE:
            wandb.log({'loss': epoch_avg_loss, 'epoch': epoch}, step=tokens_seen)
    idx2vec = model.ivectors.weight.data.cpu().numpy()
    pickle.dump(idx2vec, open(os.path.join(args.data_dir, 'idx2vec.dat'), 'wb'))
    t.save(sgns.state_dict(), os.path.join(args.save_dir, '{}.pt'.format(args.name)))
    t.save(optim.state_dict(), os.path.join(args.save_dir, '{}.optim.pt'.format(args.name)))
    
    # Finish wandb run
    if args.wandb and WANDB_AVAILABLE:
        wandb.finish()


if __name__ == '__main__':
    train(parse_args())
