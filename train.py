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
    parser.add_argument('--seed', type=int, default=42, help="random seed for reproducibility")
    parser.add_argument('--save_embeddings_limit', type=int, default=1000, help="limit number of embeddings saved to text file (0 = save all)")
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


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    t.manual_seed(seed)
    t.cuda.manual_seed_all(seed)
    # For reproducibility, but may slow down training
    t.backends.cudnn.deterministic = True
    t.backends.cudnn.benchmark = False


def train(args):
    # Set random seed for reproducibility
    set_seed(args.seed)
    print(f"Random seed set to: {args.seed}")
    
    idx2word = pickle.load(open(os.path.join(args.data_dir, 'idx2word.dat'), 'rb'))
    wc = pickle.load(open(os.path.join(args.data_dir, 'wc.dat'), 'rb'))
    wf = np.array([wc[word] for word in idx2word])
    wf = wf / wf.sum()
    ws = 1 - np.sqrt(args.ss_t / wf)
    ws = np.clip(ws, 0, 1)
    vocab_size = len(idx2word)
    weights = wf if args.weights else None
    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)
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
        config = {
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
            'seed': args.seed,
        }
        if args.torus and hasattr(sgns, 'torus_scale_factor'):
            config['torus_scale_factor'] = sgns.torus_scale_factor.item()
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_name,
            config=config
        )
    
    best_loss = float('inf')
    words_seen = 0
    
    for epoch in range(1, args.epoch + 1):
        dataset = PermutedSubsampledCorpus(os.path.join(args.data_dir, 'train.dat'))
        dataloader = DataLoader(dataset, batch_size=args.mb, shuffle=True)
        
        epoch_losses = []
        epoch_olosses = []
        epoch_nlosses = []
        pbar = tqdm(dataloader)
        pbar.set_description("[Epoch {}]".format(epoch))
        
        for batch_idx, (iword, owords) in enumerate(pbar, 1):
            oloss, nloss = sgns(iword, owords)
            loss = -(oloss + nloss).mean()
            optim.zero_grad()
            loss.backward()
            optim.step()
            
            loss_val = loss.item()
            epoch_olosses.append(oloss.item())
            epoch_nlosses.append(nloss.item())
            epoch_losses.append(loss.item())
                        
            # Count tokens: batch_size * context_size
            batch_size = iword.size(0)
            context_size = owords.size(1)
            words_seen += batch_size * context_size
            
            # Check for NaN or Inf (bug detection)
            if not np.isfinite(loss_val):
                print(f"\nWARNING: Non-finite loss detected at epoch {epoch}, batch {batch_idx}: {loss_val}")
                if args.wandb and WANDB_AVAILABLE:
                    wandb.alert(title="NaN/Inf Loss Detected", text=f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss_val}")
                break
            
            pbar.set_postfix(loss=loss_val)
            
            # Log to wandb at specified intervals
            if args.wandb and WANDB_AVAILABLE and (batch_idx % args.wandb_log_interval == 0):
                log_dict = {
                    'loss': loss_val, 
                    'epoch': epoch,
                    'avg_ologprob': np.mean(epoch_olosses),
                    'avg_nlogprob': np.mean(epoch_nlosses),
                }
                if args.torus and hasattr(sgns, 'torus_scale_factor'):
                    log_dict['torus_scale_factor'] = sgns.torus_scale_factor.item()
                if hasattr(sgns, 'coord_weights'):
                    log_dict['avg_coord_weights'] = sgns.coord_weights.mean().item()
                wandb.log(log_dict, step=words_seen)
        
        # Log at end of epoch
        epoch_avg_loss = np.mean(epoch_losses)
        if epoch_avg_loss < best_loss:
            best_loss = epoch_avg_loss
        print(f"[Epoch {epoch}/{args.epoch}] Average Loss: {epoch_avg_loss:.4f}, Best Loss: {best_loss:.4f}")
        
        # Log epoch average to wandb
        if args.wandb and WANDB_AVAILABLE:
            log_dict = {
                'loss': epoch_avg_loss, 
                'epoch': epoch,
                'avg_ologprob': np.mean(epoch_olosses),
                'avg_nlogprob': np.mean(epoch_nlosses),
            }
            if args.torus and hasattr(sgns, 'torus_scale_factor'):
                log_dict['torus_scale_factor'] = sgns.torus_scale_factor.item()
            if hasattr(sgns, 'coord_weights'):
                log_dict['avg_coord_weights'] = sgns.coord_weights.mean().item()
            wandb.log(log_dict, step=words_seen)
    idx2ivec = model.ivectors.weight.data.cpu().numpy()
    idx2ovec = model.ovectors.weight.data.cpu().numpy()
    pickle.dump(idx2ivec, open(os.path.join(args.save_dir, 'idx2ivec.dat'), 'wb'))
    pickle.dump(idx2ovec, open(os.path.join(args.save_dir, 'idx2ovec.dat'), 'wb'))
    
    # Save embeddings in readable text format: word -> embedding vector
    limit = args.save_embeddings_limit if args.save_embeddings_limit > 0 else len(idx2word)
    num_to_save = min(limit, len(idx2word))
    
    # Save input vectors (ivectors)
    ivec_txt_path = os.path.join(args.save_dir, 'embeddings_ivec.txt')
    with open(ivec_txt_path, 'w') as f:
        for idx in range(num_to_save):
            word = idx2word[idx]
            vec_str = ' '.join([str(x) for x in idx2ivec[idx]])
            f.write(f"{word} {vec_str}\n")
    print(f"Saved {num_to_save} input vector embeddings to {ivec_txt_path}")
    
    # Save output vectors (ovectors)
    ovec_txt_path = os.path.join(args.save_dir, 'embeddings_ovec.txt')
    with open(ovec_txt_path, 'w') as f:
        for idx in range(num_to_save):
            word = idx2word[idx]
            vec_str = ' '.join([str(x) for x in idx2ovec[idx]])
            f.write(f"{word} {vec_str}\n")
    print(f"Saved {num_to_save} output vector embeddings to {ovec_txt_path}")
    
    t.save(sgns.state_dict(), os.path.join(args.save_dir, '{}.pt'.format(args.name)))
    t.save(optim.state_dict(), os.path.join(args.save_dir, '{}.optim.pt'.format(args.name)))
    
    # Finish wandb run
    if args.wandb and WANDB_AVAILABLE:
        wandb.finish()


if __name__ == '__main__':
    train(parse_args())
