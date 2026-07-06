"""
MMSA - Data loader for local MOSI data
Supports both aligned and unaligned modes

Features:
- Auto-detect data format (token IDs or pre-extracted features)
- Built-in TokenToEmbedding, no HuggingFace model download required
- Automatic normalization for audio/visual features
"""
import os
import pickle
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

__all__ = ['MMDataLoader', 'MMDataset', 'TokenToEmbedding']

logger = logging.getLogger('MMSA')


# ==============================================================================
# Token-to-Embedding Converter (Built-in, no HuggingFace dependency)
# ==============================================================================
class TokenToEmbedding(nn.Module):
    """
    Converts BERT token IDs to fixed-dimension embedding vectors.
    
    Used for aligned_50.pkl and similar datasets containing raw token IDs.
    Automatically learns good text representations during training.
    """
    def __init__(self, vocab_size=30000, hidden_dim=768, seq_len=50, dropout=0.1):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.position_embedding = nn.Parameter(torch.zeros(1, seq_len, hidden_dim))
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
        # 2-layer MLP for context-aware features
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, hidden_dim)
        )
        
        nn.init.normal_(self.token_embedding.weight, mean=0, std=0.02)
        nn.init.normal_(self.position_embedding, mean=0, std=0.02)
    
    def forward(self, token_ids, attention_mask=None):
        """
        Args:
            token_ids: (batch, seq_len) BERT token IDs
        Returns:
            embeddings: (batch, seq_len, hidden_dim) BERT-like 768d features
        """
        x = self.token_embedding(token_ids)
        pos_emb = self.position_embedding[:, :x.size(1), :]
        x = x + pos_emb
        x = self.dropout(self.norm(x))
        x = self.proj(x)
        return x


# ==============================================================================
# MOSI Dataset
# ==============================================================================
class MMDataset(Dataset):
    """
    CMU-MOSI Emotion Recognition Dataset
    
    Data format (pickle file):
    {
        'train/valid/test': {
            'text': np.array (N, 3, 50) or (N, 50, 768)
            'audio': np.array (N, 50, 5)
            'vision': np.array (N, 50, 20)
            'regression_labels': np.array (N,)
        }
    }
    
    Text format auto-detection:
    - (N, 3, 50): BERT token IDs (input_ids, attention_mask, token_type_ids)
    - (N, 50, 768): Pre-extracted BERT features
    """
    def __init__(self, args, mode='train', token_to_emb=None):
        self.mode = mode
        self.args = args
        
        with open(args.featurePath, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
        
        data_split = data.get(mode, data)
        
        # ============ Text Data Loading and Format Detection ============
        raw_text = data_split.get('text')
        raw_shape = raw_text.shape
        
        if len(raw_shape) == 3 and raw_shape[1] == 3:
            # Format 1: (N, 3, 50) -> BERT token IDs
            logger.info(f"[{mode}] Detected BERT token IDs format: {raw_shape}")
            self._is_token_ids = True
            self._input_ids = raw_text[:, 0, :].astype(np.int64)
            self._token_to_emb = token_to_emb if token_to_emb else TokenToEmbedding()
            self.text_dim = 768
        else:
            # Format 2: (N, 50, 768) or other pre-extracted features
            logger.info(f"[{mode}] Pre-extracted features format: {raw_shape}")
            self._is_token_ids = False
            self.text = raw_text.astype(np.float32)
            self.text_dim = raw_text.shape[-1]
        
        # ============ Other Modalities ============
        self.audio = data_split.get('audio').astype(np.float32)
        self.vision = data_split.get('vision').astype(np.float32)
        self.labels = {
            'M': np.array(data_split.get('regression_labels')).astype(np.float32)
        }
        
        logger.info(f"[{mode}] samples: {self.labels['M'].shape}")
        logger.info(f"[{mode}] text shape: {self.text.shape if not self._is_token_ids else f'token_ids: {self._input_ids.shape}'}")
        logger.info(f"[{mode}] audio shape: {self.audio.shape}")
        logger.info(f"[{mode}] vision shape: {self.vision.shape}")
        
        # ============ Data Cleaning ============
        self.audio = np.nan_to_num(self.audio, nan=0.0)
        self.vision = np.nan_to_num(self.vision, nan=0.0)
        self.audio[self.audio == -np.inf] = 0
        self.vision[self.vision == -np.inf] = 0
        
        # ============ Normalization ============
        audio_mean = np.mean(self.audio, axis=(0, 1), keepdims=True)
        audio_std = np.std(self.audio, axis=(0, 1), keepdims=True) + 1e-8
        self.audio = (self.audio - audio_mean) / audio_std
        
        vision_mean = np.mean(self.vision, axis=(0, 1), keepdims=True)
        vision_std = np.std(self.vision, axis=(0, 1), keepdims=True) + 1e-8
        self.vision = (self.vision - vision_mean) / vision_std
    
    def __len__(self):
        return len(self.labels['M'])
    
    def get_seq_len(self):
        return (self.text.shape[1], self.audio.shape[1], self.vision.shape[1])
    
    def get_feature_dim(self):
        return (self.text_dim, self.audio.shape[2], self.vision.shape[2])
    
    def __getitem__(self, index):
        if self._is_token_ids:
            # Dynamic conversion: token IDs -> 768d embeddings
            input_ids = torch.LongTensor(self._input_ids[index:index+1])
            with torch.no_grad():
                text_feat = self._token_to_emb(input_ids)
            text_feat = text_feat.squeeze(0).numpy()
        else:
            text_feat = self.text[index]
        
        sample = {
            'text': torch.Tensor(text_feat),
            'audio': torch.Tensor(self.audio[index]),
            'vision': torch.Tensor(self.vision[index]),
            'index': index,
            'labels': {k: torch.Tensor(v[index].reshape(-1)) for k, v in self.labels.items()}
        }
        return sample
    
    def get_token_to_embedding(self):
        """Return token->embedding module for parameter registration"""
        return self._token_to_emb if self._is_token_ids else None


def MMDataLoader(args, num_workers=4):
    """
    Create MOSI data loader
    
    Args:
        args: Configuration object, must contain:
            - featurePath: Data file path
            - batch_size: Batch size
        num_workers: Number of data loading workers
    
    Returns:
        dict: {'train': DataLoader, 'valid': DataLoader, 'test': DataLoader}
    """
    # Detect if data is in token_ids format
    with open(args.featurePath, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    
    train_text = data.get('train', {}).get('text')
    is_token_ids = (len(train_text.shape) == 3 and train_text.shape[1] == 3) if train_text is not None else False
    
    if is_token_ids:
        logger.info("=" * 60)
        logger.info("Detected BERT token IDs format, using built-in TokenToEmbedding")
        logger.info("=" * 60)
        shared_token_to_emb = TokenToEmbedding()
    else:
        shared_token_to_emb = None
    
    datasets = {
        'train': MMDataset(args, mode='train', token_to_emb=shared_token_to_emb),
        'valid': MMDataset(args, mode='valid', token_to_emb=shared_token_to_emb),
        'test': MMDataset(args, mode='test', token_to_emb=shared_token_to_emb)
    }
    
    if not hasattr(args, 'seq_lens'):
        args.seq_lens = datasets['train'].get_seq_len()
    
    # Save token_to_embedding reference
    token_to_emb = shared_token_to_emb
    
    dataLoader = {}
    for ds_name in ['train', 'valid', 'test']:
        dl = DataLoader(
            datasets[ds_name],
            batch_size=args.batch_size,
            num_workers=num_workers,
            shuffle=(ds_name == 'train')
        )
        dl.token_to_embedding = token_to_emb
        dataLoader[ds_name] = dl
    
    return dataLoader
