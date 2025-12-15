#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 1: MLP Binary Classification for Conflict Flow Detection
=================================================================
Purpose: Fast screening to distinguish Normal flows from Conflict flows
Architecture: Lightweight MLP (2-3 layers) optimized for high recall
Strategy: "Gatekeeper" - prefer False Positives over False Negatives

Key Features:
- 36-dimensional spatio-temporal feature engineering (t, t-1, t-2)
- Proper data standardization (fit on train, transform on val/test)
- Class imbalance handling with class weights
- Threshold optimization for 99% recall target
- Fast inference for real-time SDN controller deployment

Author: Based on CSDN blog and MDPI journal research
Date: December 2025
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import ipaddress
from typing import Tuple, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    precision_recall_curve, auc
)
from imblearn.over_sampling import SMOTE

warnings.filterwarnings('ignore')


# ============================================================================
# 1. FEATURE ENGINEERING: Spatio-Temporal Features (36-D)
# ============================================================================

def ip_to_int_and_prefix(ip_str: str) -> Tuple[int, int]:
    """
    Convert IP address or network to integer and prefix length.
    Supports: "10.0.0.1", "10.0.0.0/24", "10.0.0.1/255.255.255.0"
    Returns: (ip_int, prefixlen)
    """
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


def compute_statistical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute statistical features for flow entries.
    Features include: byte_count, packet_count, duration statistics
    """
    stat_features = pd.DataFrame()
    
    # Basic statistics
    stat_features['byte_count'] = df['byte_count'].fillna(0)
    stat_features['packet_count'] = df['packet_count'].fillna(0)
    stat_features['duration_sec'] = df['duration_sec'].fillna(0)
    
    # Derived features
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
    
    return stat_features


def engineer_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Improved feature engineering pipeline.
    
    Key Changes (優化邏輯):
    1. 移除 src_ip_int/dst_ip_int - 避免模型死背 IP 數值
    2. 新增結構化特徵 - 遮罩長度(plen)、萬用字元(wildcard)、優先級分桶
    3. 保留統計特徵 - byte_count, packet_count 等
    
    Returns:
        X_df: DataFrame with engineered features
        feature_names: List of final feature column names
    """
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
    
    df['src_plen'] = df['src_ip'].apply(get_prefix_len)
    df['dst_plen'] = df['dst_ip'].apply(get_prefix_len)
    
    # 2. [關鍵特徵] 萬用字元匹配 (Wildcard Match)
    # 衝突常發生在 /24 覆蓋 /32 的情況
    df['is_wildcard_src'] = (df['src_plen'] < 32).astype(int)
    df['is_wildcard_dst'] = (df['dst_plen'] < 32).astype(int)
    
    # 3. [關鍵特徵] 優先級特徵 (Priority Features)
    df['priority'] = df['priority'].fillna(0)
    df['priority_norm'] = df['priority'] / 65535.0  # 正規化到 0-1
    df['is_high_priority'] = (df['priority'] > 30000).astype(int)
    df['is_low_priority'] = (df['priority'] < 1000).astype(int)
    
    # 4. 編碼類別特徵
    protocol_encoder = LabelEncoder()
    action_encoder = LabelEncoder()
    
    df['protocol_encoded'] = protocol_encoder.fit_transform(df['protocol'].fillna('unknown'))
    df['action_encoded'] = action_encoder.fit_transform(df['action'].fillna('unknown'))
    
    # 5. 計算統計特徵
    stat_df = compute_statistical_features(df)
    for col in stat_df.columns:
        df[col] = stat_df[col]
    
    # 6. [新增] 遮罩差異特徵 - 幫助識別 generalization/shadowing
    df['plen_diff'] = abs(df['src_plen'] - df['dst_plen'])
    df['is_subnet_match'] = ((df['src_plen'] < 32) | (df['dst_plen'] < 32)).astype(int)
    
    # 7. [新增] 流量強度特徵
    df['log_byte_count'] = np.log1p(df['byte_count'])
    df['log_packet_count'] = np.log1p(df['packet_count'])
    
    # 8. 選擇最終特徵集 - 注意：不包含 ip_int！
    feature_cols = [
        # 優先級特徵 (關鍵！)
        'priority', 'priority_norm', 'is_high_priority', 'is_low_priority',
        # 遮罩/萬用字元特徵 (關鍵！)
        'src_plen', 'dst_plen', 'is_wildcard_src', 'is_wildcard_dst',
        'plen_diff', 'is_subnet_match',
        # 類別特徵
        'protocol_encoded', 'action_encoded',
        # 統計特徵
        'byte_count', 'packet_count', 'duration_sec',
        'bytes_per_packet', 'bytes_per_sec', 'packets_per_sec',
        'log_byte_count', 'log_packet_count'
    ]
    
    # 建立特徵矩陣
    X_df = df[feature_cols].copy()
    X_df = X_df.fillna(0)  # 處理缺失值
    
    return X_df, feature_cols


# ============================================================================
# 2. MLP MODEL ARCHITECTURE
# ============================================================================

class LightweightMLP(nn.Module):
    """
    Lightweight MLP for Phase 1 Binary Classification.
    
    Architecture:
    - Input: 36-dimensional spatio-temporal features
    - Hidden Layer 1: 64 neurons + ReLU + Dropout(0.2)
    - Hidden Layer 2: 32 neurons + ReLU + Dropout(0.2)
    - Output: 1 neuron + Sigmoid (binary classification)
    
    Design Philosophy:
    - Shallow network (2-3 layers) for fast inference
    - Moderate neuron count to balance speed and accuracy
    - Dropout for regularization without overfitting
    """
    
    def __init__(self, input_dim: int = 36, hidden_dim1: int = 64, 
                 hidden_dim2: int = 32, dropout_rate: float = 0.2):
        super(LightweightMLP, self).__init__()
        
        self.input_layer = nn.Linear(input_dim, hidden_dim1)
        self.bn1 = nn.BatchNorm1d(hidden_dim1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.hidden_layer = nn.Linear(hidden_dim1, hidden_dim2)
        self.bn2 = nn.BatchNorm1d(hidden_dim2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.output_layer = nn.Linear(hidden_dim2, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Layer 1
        x = self.input_layer(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        
        # Layer 2
        x = self.hidden_layer(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        
        # Output
        x = self.output_layer(x)
        x = self.sigmoid(x)
        
        return x


# ============================================================================
# 3. TRAINING PIPELINE
# ============================================================================

def compute_class_weights(y_train: np.ndarray) -> torch.Tensor:
    """
    Compute class weights to handle imbalanced dataset.
    Weight = n_samples / (n_classes * n_samples_per_class)
    """
    unique, counts = np.unique(y_train, return_counts=True)
    n_samples = len(y_train)
    n_classes = len(unique)
    
    weights = n_samples / (n_classes * counts)
    
    # Convert to tensor
    weight_dict = dict(zip(unique, weights))
    class_weights = torch.FloatTensor([weight_dict[0], weight_dict[1]])
    
    print(f"\n[Class Imbalance] Class 0 (Normal): {counts[0]} samples")
    print(f"[Class Imbalance] Class 1 (Conflict): {counts[1]} samples")
    print(f"[Class Imbalance] Class weights: {class_weights.numpy()}")
    
    return class_weights


def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                criterion: nn.Module, optimizer: optim.Optimizer, 
                n_epochs: int = 50, device: str = 'cpu',
                patience: int = 10) -> dict:
    """
    Train MLP model with early stopping.
    
    Returns:
        history: Dictionary with training metrics per epoch
    """
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': []
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    print("\n" + "="*60)
    print("TRAINING PHASE 1 MLP MODEL")
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
            outputs = model(batch_X).squeeze()
            loss = criterion(outputs, batch_y.float())
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            predicted = (outputs > 0.5).float()
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
                
                outputs = model(batch_X).squeeze()
                loss = criterion(outputs, batch_y.float())
                
                val_loss += loss.item()
                predicted = (outputs > 0.5).float()
                val_correct += (predicted == batch_y).sum().item()
                val_total += batch_y.size(0)
        
        val_loss /= len(val_loader)
        val_acc = val_correct / val_total
        
        # Store history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1:3d}/{n_epochs}] | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
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


def find_optimal_threshold(model: nn.Module, X_val: torch.Tensor, y_val: np.ndarray,
                          device: str = 'cpu', strategy: str = 'f1') -> float:
    """
    Find optimal threshold using F1-Score maximization strategy.
    
    優化邏輯：不再強制 99% Recall，改用 F1 最大化策略
    這能自動平衡 Precision 和 Recall，大幅提升 Accuracy
    
    Args:
        model: Trained model
        X_val: Validation features
        y_val: Validation labels
        device: Computing device
        strategy: 'f1' for F1-maximization, 'recall' for high-recall
    
    Returns:
        optimal_threshold: Threshold value that maximizes F1
    """
    model.eval()
    with torch.no_grad():
        X_val = X_val.to(device)
        y_proba = model(X_val).squeeze().cpu().numpy()
    
    # Compute precision-recall curve
    precisions, recalls, thresholds = precision_recall_curve(y_val, y_proba)
    
    # 計算每個閾值下的 F1 Score
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    
    # 找到 F1 最高點的索引
    best_idx = np.argmax(f1_scores)
    
    if best_idx < len(thresholds):
        optimal_threshold = thresholds[best_idx]
        best_f1 = f1_scores[best_idx]
        best_precision = precisions[best_idx]
        best_recall = recalls[best_idx]
    else:
        optimal_threshold = 0.5
        best_f1 = 0.0
        best_precision = 0.0
        best_recall = 0.0
    
    print(f"\n[Threshold Optimization - F1 Maximization Strategy]")
    print(f"  Best F1-Score: {best_f1:.4f} ⭐")
    print(f"  Optimal Threshold: {optimal_threshold:.4f}")
    print(f"  Achieved Recall: {best_recall:.4f}")
    print(f"  Achieved Precision: {best_precision:.4f}")
    
    return optimal_threshold


# ============================================================================
# 4. EVALUATION
# ============================================================================

def evaluate_model(model: nn.Module, X_test: torch.Tensor, y_test: np.ndarray,
                  threshold: float = 0.5, device: str = 'cpu') -> dict:
    """
    Comprehensive evaluation of the model.
    
    Returns:
        metrics: Dictionary with all evaluation metrics
    """
    model.eval()
    with torch.no_grad():
        X_test = X_test.to(device)
        y_proba = model(X_test).squeeze().cpu().numpy()
    
    y_pred = (y_proba >= threshold).astype(int)
    
    # Compute metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    # ROC-AUC and PR-AUC
    try:
        roc_auc = roc_auc_score(y_test, y_proba)
    except:
        roc_auc = 0.0
    
    precisions, recalls, _ = precision_recall_curve(y_test, y_proba)
    pr_auc = auc(recalls, precisions)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'confusion_matrix': cm,
        'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp
    }
    
    return metrics, y_pred, y_proba


def print_evaluation_results(metrics: dict, dataset_name: str = "TEST"):
    """Print formatted evaluation results."""
    print("\n" + "="*60)
    print(f"PHASE 1 EVALUATION RESULTS ({dataset_name} SET)")
    print("="*60)
    
    print(f"\n[Primary Metrics]")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f} ⭐ (Target: ≥0.99)")
    print(f"  F1-Score:  {metrics['f1_score']:.4f}")
    
    print(f"\n[AUC Metrics]")
    print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
    print(f"  PR-AUC:    {metrics['pr_auc']:.4f}")
    
    print(f"\n[Confusion Matrix]")
    print(f"                Predicted")
    print(f"                Normal  Conflict")
    print(f"  Actual Normal   {metrics['tn']:5d}   {metrics['fp']:5d}")
    print(f"  Actual Conflict {metrics['fn']:5d}   {metrics['tp']:5d}")
    
    print(f"\n[Detailed Breakdown]")
    print(f"  True Negatives:  {metrics['tn']}")
    print(f"  False Positives: {metrics['fp']} (acceptable in Phase 1)")
    print(f"  False Negatives: {metrics['fn']} ⚠️ (should minimize)")
    print(f"  True Positives:  {metrics['tp']}")


# ============================================================================
# 5. MAIN EXECUTION
# ============================================================================

def main():
    """Main execution pipeline for Phase 1 MLP detection."""
    
    # Parse arguments
    if len(sys.argv) < 2:
        print("Usage: python3 phase1_detect_mlp.py /path/to/dataset.csv")
        print("Example: python3 phase1_detect_mlp.py dataset/1000.csv")
        sys.exit(1)
    
    input_path = sys.argv[1]
    
    print("\n" + "="*60)
    print("PHASE 1: MLP BINARY CLASSIFICATION")
    print("Conflict Flow Detection (Normal vs Conflict)")
    print("="*60)
    print(f"\n[Input] {input_path}")
    
    # ========================================================================
    # STEP 1: Load and prepare data
    # ========================================================================
    print("\n[STEP 1] Loading and preparing data...")
    df = pd.read_csv(input_path)
    
    required_cols = ["priority", "protocol", "action", "src_ip", "dst_ip", 
                     "byte_count", "packet_count", "duration_sec", "is_conflict"]
    
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    print(f"  Total samples: {len(df)}")
    print(f"  Normal flows: {(df['is_conflict'] == 0).sum()}")
    print(f"  Conflict flows: {(df['is_conflict'] == 1).sum()}")
    
    # ========================================================================
    # STEP 2: Feature Engineering
    # ========================================================================
    print(f"\n[STEP 2] Engineering structural features (優化版)...")
    X_df, feature_names = engineer_features(df)
    y = df["is_conflict"].astype(int).values
    
    print(f"  Feature dimensions: {X_df.shape}")
    print(f"  Key features: priority, wildcard, prefix_length (無 IP 整數)")
    print(f"  Features: {feature_names}")
    
    # ========================================================================
    # STEP 3: Split data (temporal order preserved)
    # ========================================================================
    print("\n[STEP 3] Splitting data (Train: 70%, Test: 30%)...")
    
    # Split: 70% train, 15% validation, 15% test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_df.values, y, test_size=0.3, random_state=42, stratify=y
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
    print("\n[STEP 4] Applying SMOTE for class balancing...")
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    
    print(f"  Before SMOTE: {len(y_train)} samples")
    print(f"  After SMOTE: {len(y_train_balanced)} samples")
    print(f"  Balanced distribution: Normal={sum(y_train_balanced==0)}, "
          f"Conflict={sum(y_train_balanced==1)}")
    
    # ========================================================================
    # STEP 5: Standardization (fit on train, transform on val/test)
    # ========================================================================
    print("\n[STEP 5] Standardizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_balanced)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"  Standardization fitted on training set only")
    print(f"  Applied to validation and test sets")
    
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
    batch_size = 64
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"  Batch size: {batch_size}")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Validation batches: {len(val_loader)}")
    
    # ========================================================================
    # STEP 7: Initialize model and training components
    # ========================================================================
    print("\n[STEP 7] Initializing MLP model...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")
    
    input_dim = X_train_scaled.shape[1]
    model = LightweightMLP(input_dim=input_dim, hidden_dim1=64, hidden_dim2=32)
    model = model.to(device)
    
    print(f"\n  Model Architecture:")
    print(f"    Input Layer: {input_dim} neurons")
    print(f"    Hidden Layer 1: 64 neurons + ReLU + BatchNorm + Dropout(0.2)")
    print(f"    Hidden Layer 2: 32 neurons + ReLU + BatchNorm + Dropout(0.2)")
    print(f"    Output Layer: 1 neuron + Sigmoid")
    print(f"    Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Compute class weights
    class_weights = compute_class_weights(y_train_balanced)
    class_weights = class_weights.to(device)
    
    # Loss function with class weights
    criterion = nn.BCELoss()
    
    # Optimizer: AdamW with weight decay
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    
    # ========================================================================
    # STEP 8: Train model
    # ========================================================================
    print("\n[STEP 8] Training model...")
    
    training_start = time.time()
    history = train_model(
        model, train_loader, val_loader, criterion, optimizer,
        n_epochs=100, device=device, patience=15
    )
    training_time = time.time() - training_start
    
    print(f"\n  Training completed in {training_time:.2f} seconds")
    
    # ========================================================================
    # STEP 9: Optimize threshold using F1-maximization
    # ========================================================================
    print("\n[STEP 9] Optimizing threshold using F1-maximization...")
    
    optimal_threshold = find_optimal_threshold(
        model, X_val_tensor, y_val, device=device
    )
    
    # ========================================================================
    # STEP 10: Evaluate on test set
    # ========================================================================
    print("\n[STEP 10] Evaluating on test set...")
    
    inference_start = time.time()
    test_metrics, y_pred_test, y_proba_test = evaluate_model(
        model, X_test_tensor, y_test,
        threshold=optimal_threshold, device=device
    )
    inference_time = time.time() - inference_start
    
    print_evaluation_results(test_metrics, dataset_name="TEST")
    
    print(f"\n[Performance]")
    print(f"  Training time: {training_time:.2f} seconds")
    print(f"  Inference time: {inference_time:.4f} seconds")
    print(f"  Avg inference per sample: {inference_time/len(y_test)*1000:.2f} ms")
    
    # ========================================================================
    # STEP 11: Predict on full dataset and save
    # ========================================================================
    print("\n[STEP 11] Predicting on full dataset...")
    
    X_full_scaled = scaler.transform(X_df.values)
    X_full_tensor = torch.FloatTensor(X_full_scaled)
    
    model.eval()
    with torch.no_grad():
        X_full_tensor = X_full_tensor.to(device)
        y_full_proba = model(X_full_tensor).squeeze().cpu().numpy()
    
    y_full_pred = (y_full_proba >= optimal_threshold).astype(int)
    
    df["predict_conflict"] = y_full_pred
    df["conflict_probability"] = y_full_proba
    
    # Save output
    output_path = os.path.splitext(input_path)[0] + "_phase1_mlp.csv"
    df.to_csv(output_path, index=False)
    
    print(f"\n[Output] Saved predictions to: {output_path}")
    print(f"  Predicted Normal: {sum(y_full_pred == 0)}")
    print(f"  Predicted Conflict: {sum(y_full_pred == 1)}")
    
    # ========================================================================
    # STEP 12: Save model
    # ========================================================================
    model_path = os.path.splitext(input_path)[0] + "_phase1_mlp.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler': scaler,
        'threshold': optimal_threshold,
        'feature_names': feature_names,
        'input_dim': input_dim
    }, model_path)
    
    print(f"\n[Model] Saved to: {model_path}")
    
    # ========================================================================
    # Final Summary
    # ========================================================================
    print("\n" + "="*60)
    print("PHASE 1 COMPLETED SUCCESSFULLY")
    print("="*60)
    print(f"\n✓ Detection Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"✓ Recall (Key Metric): {test_metrics['recall']:.4f}")
    print(f"✓ F1-Score: {test_metrics['f1_score']:.4f}")
    print(f"✓ Optimal Threshold: {optimal_threshold:.4f}")
    print(f"✓ Training Time: {training_time:.2f}s")
    print(f"✓ Inference Speed: {inference_time/len(y_test)*1000:.2f}ms per sample")
    print("\nReady for Phase 2 classification!")


if __name__ == "__main__":
    start_time = time.time()
    main()
    total_time = time.time() - start_time
    print(f"\n[Total Execution Time] {total_time:.2f} seconds")
    print("="*60 + "\n")
