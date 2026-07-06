"""
DMD Trainer - Fixed version (2026-06-21)
Complete DMD baseline training with graph distillation.
Supports both Aligned and Unaligned modes.
"""
import gc
import logging
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm


# =============================================================================
# METRICS
# =============================================================================

class MetricsTop:
    """MOSI/MOSEI metrics computation."""
    def __init__(self, train_mode='regression'):
        self.train_mode = train_mode

    def getMetics(self, dataset_name):
        if self.train_mode == "regression":
            return {
                'MOSI': self._eval_regression,
                'MOSEI': self._eval_regression,
            }.get(dataset_name.upper(), self._eval_regression)
        else:
            return self._eval_classification

    def _multiclass_acc(self, y_pred, y_true):
        return np.sum(np.round(y_pred) == np.round(y_true)) / float(len(y_true))

    def _eval_regression(self, y_pred, y_true):
        """MOSI/MOSEI regression evaluation."""
        test_preds = y_pred.view(-1).cpu().detach().numpy()
        test_truth = y_true.view(-1).cpu().detach().numpy()

        # MAE
        mae = np.mean(np.absolute(test_preds - test_truth)).astype(np.float64)

        # Corr
        corr = np.corrcoef(test_preds, test_truth)[0][1]

        # Acc7 (7-class, clipped to [-3, 3])
        test_preds_a7 = np.clip(test_preds, a_min=-3., a_max=3.)
        test_truth_a7 = np.clip(test_truth, a_min=-3., a_max=3.)
        mult_a7 = self._multiclass_acc(test_preds_a7, test_truth_a7)

        # Acc2 (binary, non-zero only)
        non_zeros = np.array([i for i, e in enumerate(test_truth) if e != 0])
        if len(non_zeros) == 0:
            non_zeros_acc2 = 0.0
        else:
            non_zeros_binary_truth = (test_truth[non_zeros] > 0)
            non_zeros_binary_preds = (test_preds[non_zeros] > 0)
            non_zeros_acc2 = np.mean(non_zeros_binary_preds == non_zeros_binary_truth)

        # F1 (non-zero binary)
        tp = np.sum((non_zeros_binary_preds == 1) & (non_zeros_binary_truth == 1))
        fp = np.sum((non_zeros_binary_preds == 1) & (non_zeros_binary_truth == 0))
        fn = np.sum((non_zeros_binary_preds == 0) & (non_zeros_binary_truth == 1))
        f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0

        return {
            "Acc_2": round(float(non_zeros_acc2), 4),
            "F1_score": round(float(f1), 4),
            "Acc_7": round(float(mult_a7), 4),
            "MAE": round(float(mae), 4),
            "Corr": round(float(corr), 4) if not np.isnan(corr) else 0.0,
        }


# =============================================================================
# DMD TRAINER
# =============================================================================

class DMDTrainer:
    """
    DMD model trainer (CVPR 2023 baseline).
    Handles both Aligned and Unaligned modes with graph distillation.
    
    Components:
    - DMD backbone model
    - Homo GD kernel (per-modality distillation)
    - Hetero GD kernel (cross-modality distillation)
    """
    def __init__(self, model, dataset, config, logger=None):
        """
        Initialize DMD trainer.
        
        Args:
            model: DMD model instance
            dataset: Dataset with 'train', 'valid', 'test' splits
            config: DMDConfig instance
            logger: Optional logger
        """
        self.model = model
        self.dataset = dataset
        self.config = config
        self.logger = logger or logging.getLogger('MMSA')
        
        # Determine device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Loss function
        self.criterion = nn.L1Loss()
        
        # Metrics
        self.metrics = MetricsTop(train_mode='regression').getMetics(config.dataset_name.upper())
        
        # Optimizer with all trainable parameters
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # Training state
        self.best_valid = None
        self.best_epoch = 0
        self.epochs = 0
        
        # Create checkpoint directory
        self.pt_dir = Path("./pt")
        self.pt_dir.mkdir(parents=True, exist_ok=True)
        
    def _dict_to_str(self, results):
        """Format metrics dict to string."""
        return ' '.join([f"{k}: {v:.4f}" for k, v in results.items()])
    
    def train_epoch(self, dataloader):
        """Train one epoch."""
        self.model.train()
        y_pred, y_true = [], []
        train_loss = 0.0
        num_batches = 0
        
        with tqdm(dataloader, desc=f"Epoch {self.epochs}") as td:
            for batch_data in td:
                # Get data
                text = batch_data['text'].to(self.device)
                audio = batch_data['audio'].to(self.device)
                vision = batch_data['vision'].to(self.device)
                labels = batch_data['labels']['M'].to(self.device).view(-1, 1)
                
                # Forward pass
                self.optimizer.zero_grad()
                output = self.model(text, audio, vision, is_distill=True)
                
                # Main task loss
                loss = self.criterion(output['output_logit'], labels.squeeze())
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                if hasattr(self.config, 'grad_clip') and self.config.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                
                self.optimizer.step()
                
                # Accumulate metrics
                train_loss += loss.item()
                y_pred.append(output['output_logit'].cpu())
                y_true.append(labels.cpu())
                num_batches += 1
                
                td.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Compute epoch metrics
        train_loss /= num_batches
        pred, true = torch.cat(y_pred), torch.cat(y_true)
        train_results = self.metrics(pred, true)
        train_results['Loss'] = round(train_loss, 4)
        
        return train_results
    
    def evaluate(self, dataloader, mode="VAL"):
        """Evaluate on given dataloader."""
        self.model.eval()
        y_pred, y_true = [], []
        eval_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            with tqdm(dataloader, desc=f"{mode}") as td:
                for batch_data in td:
                    text = batch_data['text'].to(self.device)
                    audio = batch_data['audio'].to(self.device)
                    vision = batch_data['vision'].to(self.device)
                    labels = batch_data['labels']['M'].to(self.device).view(-1, 1)
                    
                    output = self.model(text, audio, vision, is_distill=False)
                    loss = self.criterion(output['output_logit'], labels.squeeze())
                    
                    eval_loss += loss.item()
                    y_pred.append(output['output_logit'].cpu())
                    y_true.append(labels.cpu())
                    num_batches += 1
        
        eval_loss /= num_batches
        pred, true = torch.cat(y_pred), torch.cat(y_true)
        eval_results = self.metrics(pred, true)
        eval_results['Loss'] = round(eval_loss, 4)
        
        return eval_results
    
    def train(self):
        """
        Main training loop.
        
        Returns:
            dict: Training results with metrics
        """
        self.logger.info("=" * 60)
        self.logger.info(f"DMD Training Started")
        self.logger.info(f"Mode: {self.config.mode}, Seed: {self.config.seed}")
        self.logger.info(f"Device: {self.device}")
        self.logger.info("=" * 60)
        
        dataloaders = {
            'train': self.dataset.get_dataloader('train', self.config.batch_size),
            'valid': self.dataset.get_dataloader('valid', self.config.batch_size),
            'test': self.dataset.get_dataloader('test', self.config.batch_size),
        }
        
        best_valid = float('inf')
        best_test = None
        patience_counter = 0
        
        for epoch in range(1, self.config.epochs + 1):
            self.epochs = epoch
            
            # Train
            train_results = self.train_epoch(dataloaders['train'])
            self.logger.info(
                f">> Epoch {epoch} TRAIN [{epoch - self.best_epoch}/{epoch}] "
                f"loss: {train_results['Loss']:.4f} {self._dict_to_str(train_results)}"
            )
            
            # Validate
            val_results = self.evaluate(dataloaders['valid'], mode="VAL")
            self.logger.info(f"   VAL >> {self._dict_to_str(val_results)}")
            
            # Update scheduler
            self.scheduler.step(val_results['Loss'])
            
            # Test
            test_results = self.evaluate(dataloaders['test'], mode="TEST")
            self.logger.info(f"   TEST >> {self._dict_to_str(test_results)}")
            
            # Save checkpoint
            torch.save(self.model.state_dict(), self.pt_dir / f"epoch_{epoch}.pth")
            
            # Check improvement
            is_better = val_results['Loss'] < best_valid - 1e-6
            if is_better:
                best_valid = val_results['Loss']
                best_test = test_results
                self.best_epoch = epoch
                patience_counter = 0
                torch.save(self.model.state_dict(), self.pt_dir / "best_dmd.pth")
                self.logger.info(f"   >> New best model saved (epoch {epoch})")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= getattr(self.config, 'early_stop', 20):
                self.logger.info(f">> Early stopping at epoch {epoch}")
                break
            
            self.logger.info("-" * 60)
        
        # Final results
        self.logger.info("=" * 60)
        self.logger.info(f"Training Complete")
        self.logger.info(f"Best epoch: {self.best_epoch}")
        self.logger.info(f"Best VALID Loss: {best_valid:.4f}")
        self.logger.info(f"Best TEST: {self._dict_to_str(best_test)}")
        self.logger.info("=" * 60)
        
        return {
            'best_epoch': self.best_epoch,
            'best_valid_loss': best_valid,
            'best_test': best_test,
            'train_history': None  # TODO: add history tracking
        }
    
    def load_best_model(self):
        """Load the best model checkpoint."""
        best_path = self.pt_dir / "best_dmd.pth"
        if best_path.exists():
            self.model.load_state_dict(torch.load(best_path, map_location=self.device))
            self.logger.info(f"Loaded best model from {best_path}")
        else:
            self.logger.warning(f"Best model not found at {best_path}")
    
    def get_model(self):
        """Get the model instance."""
        return self.model


# =============================================================================
# Standalone training functions
# =============================================================================

def train_dmd_aligned(seed=42):
    """Train DMD on aligned data."""
    from config import DMDConfig
    from data.dataset import MOSEIDataset
    from models.dmd_model import DMD
    
    config = DMDConfig.create_aligned_config(seed)
    dataset = MOSEIDataset(config)
    model = DMD(config)
    trainer = DMDTrainer(model, dataset, config)
    
    return trainer.train()


def train_dmd_unaligned(seed=42):
    """Train DMD on unaligned data."""
    from config import DMDConfig
    from data.dataset import MOSEIDataset
    from models.dmd_model import DMD
    
    config = DMDConfig.create_unaligned_config(seed)
    dataset = MOSEIDataset(config)
    model = DMD(config)
    trainer = DMDTrainer(model, dataset, config)
    
    return trainer.train()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='DMD Training')
    parser.add_argument('--mode', type=str, default='aligned', choices=['aligned', 'unaligned'])
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    if args.mode == 'aligned':
        train_dmd_aligned(args.seed)
    else:
        train_dmd_unaligned(args.seed)
