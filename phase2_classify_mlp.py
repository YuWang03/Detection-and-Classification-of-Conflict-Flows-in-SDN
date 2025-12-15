#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 2: MLP Multiclass Classification for Conflict Flow Types
================================================================
Purpose: Precise diagnosis of conflict types (7 categories)
Architecture: Deeper MLP (4-5 layers) with Focal Loss for handling class imbalance
Strategy: "Expert" - focus on high precision and macro-F1 score

Conflict Types:
1. Redundancy
2. Shadowing
3. Overlapping
4. Correlation A
5. Correlation B
6. Generalisation
7. Imbrication

Key Features:
- Only processes flows identified as conflicts in Phase 1
- Deeper architecture (4-5 layers, 128/256 neurons) for complex decision boundaries
- Focal Loss to handle long-tail distribution (rare attack types)
- AdamW optimizer with Cosine Annealing learning rate scheduler
- Emphasis on Macro-F1 and Macro-Precision metrics
- Dropout and Batch Normalization for regularization

Author: Based on CSDN blog and MDPI journal research
Date: December 2025
"""

import os
import sys
import time
import warnings
import argparse
import numpy as np
import pandas as pd
import ipaddress
from typing import Tuple, List, Dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from imblearn.over_sampling import SMOTE, ADASYN

warnings.filterwarnings('ignore')


# ============================================================================
# 1. FOCAL LOSS FOR HANDLING CLASS IMBALANCE
# ============================================================================

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance in multiclass classification.
    
    Formula: FL(p_t) = -α(1 - p_t)^γ * log(p_t)
    
    Args:
        gamma: Focusing parameter (default: 2.0, reduced for better generalization)
               Higher gamma means more focus on hard examples
        alpha: Class weighting factor (default: None, will be computed)
        reduction: 'mean' or 'sum'
        label_smoothing: Label smoothing factor (default: 0.1)
    
    Reference: Lin et al. "Focal Loss for Dense Object Detection" (2017)
    """
    
    def __init__(self, gamma: float = 2.0, alpha=None, reduction: str = 'mean', label_smoothing: float = 0.1):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.label_smoothing = label_smoothing
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Model predictions (logits) of shape [batch_size, num_classes]
            targets: Ground truth labels of shape [batch_size]
        
        Returns:
            Focal loss value with label smoothing
        """
        # Convert logits to probabilities
        probs = torch.softmax(inputs, dim=1)
        num_classes = inputs.shape[1]
        
        # Apply label smoothing
        targets_one_hot = torch.zeros_like(probs)
        targets_one_hot.scatter_(1, targets.unsqueeze(1), 1)
        
        if self.label_smoothing > 0:
            targets_one_hot = targets_one_hot * (1 - self.label_smoothing) + \
                             self.label_smoothing / num_classes
        
        p_t = (probs * targets_one_hot).sum(dim=1)
        
        # Compute focal loss
        focal_weight = (1 - p_t) ** self.gamma
        
        # Cross entropy loss
        ce_loss = -torch.log(p_t + 1e-8)
        
        # Apply focal weight
        loss = focal_weight * ce_loss
        
        # Apply alpha weighting if provided
        if self.alpha is not None:
            if isinstance(self.alpha, (list, np.ndarray)):
                alpha_t = torch.tensor(self.alpha, device=inputs.device)[targets]
                loss = alpha_t * loss
        
        # Reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


# ============================================================================
# 2. FEATURE ENGINEERING FOR PHASE 2
# ============================================================================

def ip_to_int_and_prefix(ip_str: str) -> Tuple[int, int]:
    """Convert IP address or network to integer and prefix length."""
    if pd.isna(ip_str):
        return 0, 0
    s = str(ip_str).strip()
    
    try:
        if "/" in s:
            net = ipaddress.ip_network(s, strict=False)
            return int(net.network_address), int(net.prefixlen)
        else:
            ip = ipaddress.ip_address(s)
            return int(ip), 32
    except Exception:
        return 0, 0


def build_structural_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build structural features for conflict type classification.
    
    優化邏輯：移除 IP 整數，加入結構化特徵
    衝突的本質是 Match Fields 的重疊與 Priority 的競賽
    """
    features = pd.DataFrame(index=df.index)
    
    # 1. 解析 Prefix Length (遮罩長度) - 不使用 IP 整數！
    def get_prefix_len(ip_str):
        if pd.isna(ip_str):
            return 32
        s = str(ip_str).strip()
        if '/' in s:
            try:
                return int(s.split('/')[1])
            except:
                return 32
        return 32
    
    features['src_plen'] = df['src_ip'].apply(get_prefix_len)
    features['dst_plen'] = df['dst_ip'].apply(get_prefix_len)
    
    # 2. [關鍵特徵] 萬用字元匹配 (Wildcard Match)
    features['is_wildcard_src'] = (features['src_plen'] < 32).astype(int)
    features['is_wildcard_dst'] = (features['dst_plen'] < 32).astype(int)
    
    # 3. [關鍵特徵] 遮罩差異 - 幫助識別 generalization/shadowing
    features['plen_diff'] = abs(features['src_plen'] - features['dst_plen'])
    features['is_subnet_match'] = ((features['src_plen'] < 32) | (features['dst_plen'] < 32)).astype(int)
    
    # 4. 網段範圍特徵 (以 2 的冪次表示，避免數值過大)
    features['src_range_log'] = 32 - features['src_plen']  # log2 of range size
    features['dst_range_log'] = 32 - features['dst_plen']
    
    return features


def engineer_classification_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Improved feature engineering pipeline for Phase 2 classification.
    
    優化邏輯：
    1. 移除 IP 整數 (ip_int) - 避免過擬合
    2. 加入結構化特徵 - plen, wildcard, priority bucket
    3. 保留有用的統計特徵
    
    Returns:
        X_df: DataFrame with engineered features
        feature_names: List of feature column names
    """
    # 1. 結構化特徵 (不使用 IP 整數！)
    struct_features = build_structural_features(df)
    
    # 2. 優先級特徵 (關鍵！)
    priority_features = pd.DataFrame({
        'priority': df['priority'].fillna(0),
        'priority_norm': df['priority'].fillna(0) / 65535.0,
        'is_high_priority': (df['priority'].fillna(0) > 30000).astype(int),
        'is_low_priority': (df['priority'].fillna(0) < 1000).astype(int),
        'priority_log': np.log1p(df['priority'].fillna(0))
    }, index=df.index)
    
    # 3. 編碼類別特徵
    protocol_encoder = LabelEncoder()
    action_encoder = LabelEncoder()
    
    categorical_features = pd.DataFrame({
        'protocol_encoded': protocol_encoder.fit_transform(df['protocol'].fillna('unknown')),
        'action_encoded': action_encoder.fit_transform(df['action'].fillna('unknown'))
    }, index=df.index)
    
    # 4. 統計特徵 (簡化版)
    stat_features = pd.DataFrame({
        'byte_count': df['byte_count'].fillna(0),
        'packet_count': df['packet_count'].fillna(0),
        'duration_sec': df['duration_sec'].fillna(0),
    }, index=df.index)
    
    # 衍生統計特徵
    stat_features['bytes_per_packet'] = np.where(
        stat_features['packet_count'] > 0,
        stat_features['byte_count'] / stat_features['packet_count'],
        0
    )
    
    stat_features['bytes_per_sec'] = np.where(
        stat_features['duration_sec'] > 0,
        stat_features['byte_count'] / stat_features['duration_sec'],
        0
    )
    
    stat_features['packets_per_sec'] = np.where(
        stat_features['duration_sec'] > 0,
        stat_features['packet_count'] / stat_features['duration_sec'],
        0
    )
    
    # Log 轉換
    stat_features['log_byte_count'] = np.log1p(stat_features['byte_count'])
    stat_features['log_packet_count'] = np.log1p(stat_features['packet_count'])
    
    # 5. 合併所有特徵
    X_df = pd.concat([
        struct_features,
        priority_features,
        categorical_features,
        stat_features
    ], axis=1)
    
    feature_names = X_df.columns.tolist()
    
    return X_df, feature_names


# ============================================================================
# 3. RESNET-STYLE MLP MODEL ARCHITECTURE
# ============================================================================

class ResNetMLP(nn.Module):
    """
    Wide Residual MLP for better tabular data performance.
    
    優化邏輯：
    1. 加寬而非加深 - 3層但更寬
    2. 殘差連接 (Skip Connection) - 讓訓練更穩定
    3. 適度的 Dropout - 防止過擬合
    
    Architecture:
    - Input Block: input_dim -> hidden_dim (256)
    - Residual Block: hidden_dim -> hidden_dim (with skip connection)
    - Output Layer: hidden_dim -> num_classes
    """
    
    def __init__(self, input_dim: int, num_classes: int = 7, hidden_dim: int = 256):
        super(ResNetMLP, self).__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        
        # Input Block: project to hidden_dim
        self.input_block = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3)
        )
        
        # Residual Block 1
        self.res_block1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3)
        )
        
        # Residual Block 2
        self.res_block2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2)
        )
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        # Input projection
        out = self.input_block(x)
        
        # Residual Block 1 with skip connection
        residual = out
        out = self.res_block1(out)
        out = out + residual  # Skip connection
        
        # Residual Block 2 with skip connection
        residual = out
        out = self.res_block2(out)
        out = out + residual  # Skip connection
        
        # Output
        out = self.output_layer(out)
        return out


# Alias for backward compatibility
DeepMLP = ResNetMLP


# ============================================================================
# 4. TRAINING PIPELINE WITH COSINE ANNEALING
# ============================================================================

def train_model_with_scheduler(model: nn.Module, train_loader: DataLoader,
                                val_loader: DataLoader, criterion: nn.Module,
                                optimizer: optim.Optimizer, scheduler,
                                n_epochs: int = 100, device: str = 'cpu',
                                patience: int = 15) -> dict:
    """
    Train Deep MLP with Cosine Annealing learning rate scheduler.
    
    Returns:
        history: Dictionary with training metrics per epoch
    """
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'learning_rates': []
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    print("\n" + "="*60)
    print("TRAINING PHASE 2 DEEP MLP MODEL")
    print("="*60)
    
    for epoch in range(n_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_correct += (predicted == batch_y).sum().item()
            train_total += batch_y.size(0)
        
        train_loss /= len(train_loader)
        train_acc = train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == batch_y).sum().item()
                val_total += batch_y.size(0)
        
        val_loss /= len(val_loader)
        val_acc = val_correct / val_total
        
        # Step scheduler
        if scheduler is not None:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
        else:
            current_lr = optimizer.param_groups[0]['lr']
        
        # Store history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['learning_rates'].append(current_lr)
        
        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1:3d}/{n_epochs}] | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
                  f"LR: {current_lr:.6f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n[Early Stopping] No improvement for {patience} epochs.")
                break
    
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\n[Model] Restored best model with val_loss: {best_val_loss:.4f}")
    
    return history


# ============================================================================
# 5. EVALUATION WITH MACRO METRICS
# ============================================================================

def evaluate_multiclass_model(model: nn.Module, X_test: torch.Tensor,
                               y_test: np.ndarray, class_names: List[str],
                               device: str = 'cpu') -> dict:
    """
    Comprehensive evaluation for multiclass classification.
    Focus on Macro-F1 and Macro-Precision for balanced evaluation.
    
    Returns:
        metrics: Dictionary with all evaluation metrics
    """
    model.eval()
    with torch.no_grad():
        X_test = X_test.to(device)
        outputs = model(X_test)
        _, y_pred = torch.max(outputs, 1)
        y_pred = y_pred.cpu().numpy()
    
    # Compute metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    # Macro metrics (equal weight to each class, good for imbalanced data)
    macro_precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
    macro_recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
    macro_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    
    # Weighted metrics (weighted by support)
    weighted_precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    weighted_recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    weighted_f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    # Per-class metrics
    per_class_precision = precision_score(y_test, y_pred, average=None, zero_division=0)
    per_class_recall = recall_score(y_test, y_pred, average=None, zero_division=0)
    per_class_f1 = f1_score(y_test, y_pred, average=None, zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    metrics = {
        'accuracy': accuracy,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'weighted_precision': weighted_precision,
        'weighted_recall': weighted_recall,
        'weighted_f1': weighted_f1,
        'per_class_precision': per_class_precision,
        'per_class_recall': per_class_recall,
        'per_class_f1': per_class_f1,
        'confusion_matrix': cm,
        'y_pred': y_pred
    }
    
    return metrics


def print_classification_results(metrics: dict, class_names: List[str],
                                 dataset_name: str = "TEST"):
    """Print formatted classification results."""
    print("\n" + "="*60)
    print(f"PHASE 2 CLASSIFICATION RESULTS ({dataset_name} SET)")
    print("="*60)
    
    print(f"\n[Overall Metrics]")
    print(f"  Accuracy:         {metrics['accuracy']:.4f}")
    print(f"  Macro-Precision:  {metrics['macro_precision']:.4f} ⭐")
    print(f"  Macro-Recall:     {metrics['macro_recall']:.4f}")
    print(f"  Macro-F1:         {metrics['macro_f1']:.4f} ⭐")
    
    print(f"\n[Weighted Metrics]")
    print(f"  Weighted Precision: {metrics['weighted_precision']:.4f}")
    print(f"  Weighted Recall:    {metrics['weighted_recall']:.4f}")
    print(f"  Weighted F1:        {metrics['weighted_f1']:.4f}")
    
    print(f"\n[Per-Class Performance]")
    print(f"{'Class':<20} {'Precision':>10} {'Recall':>10} {'F1-Score':>10}")
    print("-" * 55)
    for i, class_name in enumerate(class_names):
        if i < len(metrics['per_class_precision']):
            print(f"{class_name:<20} {metrics['per_class_precision'][i]:>10.4f} "
                  f"{metrics['per_class_recall'][i]:>10.4f} "
                  f"{metrics['per_class_f1'][i]:>10.4f}")
    
    print(f"\n[Confusion Matrix]")
    cm = metrics['confusion_matrix']
    
    # Print header
    print(f"{'':>15}", end="")
    for name in class_names:
        print(f"{name[:8]:>10}", end="")
    print()
    
    # Print matrix
    for i, name in enumerate(class_names):
        print(f"{name:<15}", end="")
        for j in range(len(class_names)):
            if i < cm.shape[0] and j < cm.shape[1]:
                print(f"{cm[i, j]:>10}", end="")
        print()


# ============================================================================
# 6. MAIN EXECUTION
# ============================================================================

def main():
    """Main execution pipeline for Phase 2 MLP classification."""
    
    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Phase 2: MLP Multiclass Classification for Conflict Types'
    )
    parser.add_argument('--input', type=str, required=False,
                       help='Path to Phase 1 output CSV (e.g., 1000_phase1_mlp.csv)')
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='Test set size (default: 0.2)')
    parser.add_argument('--epochs', type=int, default=200,
                       help='Number of training epochs (default: 200)')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size (default: 64)')
    parser.add_argument('--use_smote', action='store_true', default=True,
                       help='Use SMOTE for oversampling (default: True)')
    parser.add_argument('--learning_rate', type=float, default=0.002,
                       help='Initial learning rate (default: 0.002)')
    
    args = parser.parse_args()
    
    # Handle input
    if args.input:
        input_path = args.input
    elif len(sys.argv) >= 2 and not sys.argv[1].startswith('--'):
        input_path = sys.argv[1]
    else:
        print("Usage: python3 phase2_classify_mlp.py <input_csv> [options]")
        print("Example: python3 phase2_classify_mlp.py dataset/1000_phase1_mlp.csv")
        print("\nOr use: python3 phase2_classify_mlp.py --input <path> [--epochs 100] [--use_smote]")
        sys.exit(1)
    
    print("\n" + "="*60)
    print("PHASE 2: MLP MULTICLASS CLASSIFICATION")
    print("Conflict Type Classification (7 Types)")
    print("="*60)
    print(f"\n[Input] {input_path}")
    
    # ========================================================================
    # STEP 1: Load data and filter conflicts only
    # ========================================================================
    print("\n[STEP 1] Loading data and filtering conflict flows...")
    df = pd.read_csv(input_path)
    
    # Filter only conflict flows (predicted or actual)
    if 'predict_conflict' in df.columns:
        df_conflict = df[df['predict_conflict'] == 1].copy()
        print(f"  Using 'predict_conflict' column for filtering")
    else:
        df_conflict = df[df['is_conflict'] == 1].copy()
        print(f"  Using 'is_conflict' column for filtering")
    
    if len(df_conflict) == 0:
        print("\n[Error] No conflict flows found in input data!")
        print("Make sure to run Phase 1 detection first.")
        sys.exit(1)
    
    print(f"  Total conflict flows: {len(df_conflict)}")
    
    # Normalize conflict types
    df_conflict['conflict_type'] = df_conflict['conflict_type'].apply(
        lambda x: str(x).strip().lower()
    )
    
    # Count conflict types
    conflict_distribution = df_conflict['conflict_type'].value_counts()
    print(f"\n  Conflict type distribution:")
    for conflict_type, count in conflict_distribution.items():
        print(f"    {conflict_type}: {count}")
    
    # ========================================================================
    # STEP 2: Feature Engineering
    # ========================================================================
    print("\n[STEP 2] Engineering features for classification...")
    X_df, feature_names = engineer_classification_features(df_conflict)
    
    # Encode target labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df_conflict['conflict_type'].values)
    class_names = label_encoder.classes_
    num_classes = len(class_names)
    
    print(f"  Feature dimensions: {X_df.shape}")
    print(f"  Number of classes: {num_classes}")
    print(f"  Class names: {list(class_names)}")
    
    # ========================================================================
    # STEP 3: Split data
    # ========================================================================
    print("\n[STEP 3] Splitting data...")
    
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_df.values, y, test_size=args.test_size, random_state=42, stratify=y
    )
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"  Train set: {len(X_train)} samples")
    print(f"  Validation set: {len(X_val)} samples")
    print(f"  Test set: {len(X_test)} samples")
    
    # ========================================================================
    # STEP 4: Handle class imbalance with SMOTE
    # ========================================================================
    if args.use_smote:
        print("\n[STEP 4] Applying SMOTE for class balancing...")
        try:
            # Use SMOTE with more neighbors for better synthesis
            smote = SMOTE(random_state=42, k_neighbors=min(5, len(y_train)//num_classes - 1))
            X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
            print(f"  Before SMOTE: {len(y_train)} samples")
            print(f"  After SMOTE: {len(y_train_balanced)} samples")
            
            # Show class distribution after SMOTE
            unique, counts = np.unique(y_train_balanced, return_counts=True)
            print(f"  Class distribution after SMOTE:")
            for cls, cnt in zip(unique, counts):
                print(f"    Class {class_names[cls]}: {cnt} samples")
        except Exception as e:
            print(f"  [Warning] SMOTE failed: {e}")
            print(f"  Using original training data")
            X_train_balanced, y_train_balanced = X_train, y_train
    else:
        print("\n[STEP 4] Skipping oversampling")
        X_train_balanced, y_train_balanced = X_train, y_train
    
    # ========================================================================
    # STEP 5: Standardization
    # ========================================================================
    print("\n[STEP 5] Standardizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_balanced)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"  Fitted StandardScaler on training set")
    
    # ========================================================================
    # STEP 6: Convert to PyTorch tensors
    # ========================================================================
    print("\n[STEP 6] Converting to PyTorch tensors...")
    
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    y_train_tensor = torch.LongTensor(y_train_balanced)
    X_val_tensor = torch.FloatTensor(X_val_scaled)
    y_val_tensor = torch.LongTensor(y_val)
    X_test_tensor = torch.FloatTensor(X_test_scaled)
    y_test_tensor = torch.LongTensor(y_test)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    print(f"  Batch size: {args.batch_size}")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Validation batches: {len(val_loader)}")
    
    # ========================================================================
    # STEP 7: Initialize ResNet-style MLP model
    # ========================================================================
    print("\n[STEP 7] Initializing ResNet-style MLP model...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")
    
    input_dim = X_train_scaled.shape[1]
    model = ResNetMLP(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dim=256  # Wide but shallow
    )
    model = model.to(device)
    
    print(f"\n  Model Architecture (ResNet-style):")
    print(f"    Input Block: {input_dim} -> 256 neurons + LeakyReLU + BN + Dropout(0.3)")
    print(f"    Residual Block 1: 256 -> 256 (with skip connection)")
    print(f"    Residual Block 2: 256 -> 256 (with skip connection)")
    print(f"    Output Layer: 256 -> {num_classes} neurons")
    print(f"    Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # ========================================================================
    # STEP 8: Setup Focal Loss and optimizer
    # ========================================================================
    print("\n[STEP 8] Setting up Focal Loss and AdamW optimizer...")
    
    # Focal Loss with label smoothing for better generalization
    criterion = FocalLoss(gamma=2.0, reduction='mean', label_smoothing=0.1)
    print(f"  Using Focal Loss (gamma=2.0, label_smoothing=0.1)")
    
    # AdamW optimizer with better learning rate
    lr = args.learning_rate if hasattr(args, 'learning_rate') else 0.002
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.005, betas=(0.9, 0.999))
    print(f"  Using AdamW optimizer (lr={lr}, weight_decay=0.005)")
    
    # Cosine Annealing with Warm Restarts - longer cycles
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=1e-6
    )
    print(f"  Using CosineAnnealingWarmRestarts scheduler (T_0=20)")
    
    # ========================================================================
    # STEP 9: Train model
    # ========================================================================
    print(f"\n[STEP 9] Training model for {args.epochs} epochs...")
    
    training_start = time.time()
    history = train_model_with_scheduler(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        n_epochs=args.epochs, device=device, patience=30
    )
    training_time = time.time() - training_start
    
    print(f"\n  Training completed in {training_time:.2f} seconds")
    
    # ========================================================================
    # STEP 10: Evaluate on test set
    # ========================================================================
    print("\n[STEP 10] Evaluating on test set...")
    
    inference_start = time.time()
    test_metrics = evaluate_multiclass_model(
        model, X_test_tensor, y_test, class_names, device=device
    )
    inference_time = time.time() - inference_start
    
    print_classification_results(test_metrics, class_names, dataset_name="TEST")
    
    print(f"\n[Performance]")
    print(f"  Training time: {training_time:.2f} seconds")
    print(f"  Inference time: {inference_time:.4f} seconds")
    print(f"  Avg inference per sample: {inference_time/len(y_test)*1000:.2f} ms")
    
    # ========================================================================
    # STEP 11: Predict on full conflict dataset
    # ========================================================================
    print("\n[STEP 11] Predicting on full conflict dataset...")
    
    X_full_scaled = scaler.transform(X_df.values)
    X_full_tensor = torch.FloatTensor(X_full_scaled)
    
    model.eval()
    with torch.no_grad():
        X_full_tensor = X_full_tensor.to(device)
        outputs = model(X_full_tensor)
        _, y_full_pred = torch.max(outputs, 1)
        y_full_pred = y_full_pred.cpu().numpy()
    
    # Decode predictions
    df_conflict['predicted_conflict_type'] = label_encoder.inverse_transform(y_full_pred)
    
    # Save output
    output_path = os.path.splitext(input_path)[0] + "_phase2_mlp.csv"
    df_conflict.to_csv(output_path, index=False)
    
    print(f"\n[Output] Saved predictions to: {output_path}")
    pred_distribution = df_conflict['predicted_conflict_type'].value_counts()
    print(f"\n  Predicted conflict type distribution:")
    for conflict_type, count in pred_distribution.items():
        print(f"    {conflict_type}: {count}")
    
    # ========================================================================
    # STEP 12: Save model
    # ========================================================================
    model_path = os.path.splitext(input_path)[0] + "_phase2_mlp.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler': scaler,
        'label_encoder': label_encoder,
        'class_names': class_names,
        'feature_names': feature_names,
        'input_dim': input_dim,
        'num_classes': num_classes
    }, model_path)
    
    print(f"\n[Model] Saved to: {model_path}")
    
    # ========================================================================
    # Final Summary
    # ========================================================================
    print("\n" + "="*60)
    print("PHASE 2 COMPLETED SUCCESSFULLY")
    print("="*60)
    print(f"\n✓ Classification Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"✓ Macro-F1 Score: {test_metrics['macro_f1']:.4f}")
    print(f"✓ Macro-Precision: {test_metrics['macro_precision']:.4f}")
    print(f"✓ Training Time: {training_time:.2f}s")
    print(f"✓ Inference Speed: {inference_time/len(y_test)*1000:.2f}ms per sample")
    print("\nConflict type classification complete!")


if __name__ == "__main__":
    start_time = time.time()
    main()
    total_time = time.time() - start_time
    print(f"\n[Total Execution Time] {total_time:.2f} seconds")
    print("="*60 + "\n")
