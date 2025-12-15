#!/usr/bin/env python3
import os
import sys
import time
import ipaddress
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from xgboost import XGBClassifier


def ip_to_int_and_prefix(ip_str: str):
    """
    Supports:
      - "10.0.0.1"
      - "10.0.0.0/24"
      - "10.0.0.1/255.255.255.0"
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


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 phase1_detection_XGB.py /path/to/1000.csv")
        sys.exit(1)

    in_path = sys.argv[1]
    df = pd.read_csv(in_path)

    required = ["priority", "protocol", "action", "src_ip", "dst_ip", "is_conflict"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")

    # ---- feature engineering for addr ----
    src_parsed = df["src_ip"].apply(ip_to_int_and_prefix)
    dst_parsed = df["dst_ip"].apply(ip_to_int_and_prefix)

    df["src_ip_int"] = [x[0] for x in src_parsed]
    df["src_plen"] = [x[1] for x in src_parsed]
    df["dst_ip_int"] = [x[0] for x in dst_parsed]
    df["dst_plen"] = [x[1] for x in dst_parsed]

    # ---- Phase1 features (priority, protocol, actions, addr) ----
    X = df[["priority", "protocol", "action", "src_ip_int", "src_plen", "dst_ip_int", "dst_plen"]]
    y = df["is_conflict"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["protocol", "action"]),
            ("num", "passthrough", ["priority", "src_ip_int", "src_plen", "dst_ip_int", "dst_plen"]),
        ]
    )

    # ---- handle imbalance (conflict 少) ----
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale_pos_weight = (neg / pos) if pos > 0 else 1.0

    clf = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        min_child_weight=1,
        gamma=0,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight,
    )

    model = Pipeline([("pre", pre), ("clf", clf)])
    model.fit(X_train, y_train)

    # ---- evaluation on test set ----
    y_pred_test = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred_test)

    print(f"\n[Phase1-XGB] Input : {in_path}")
    print("\n[Phase1-XGB] Metrics on TEST set (predict_conflict vs is_conflict)")
    print(classification_report(y_test, y_pred_test, digits=4, zero_division=0))

    print("[Phase1-XGB] Confusion Matrix (rows=true, cols=pred) => [[TN FP],[FN TP]]")
    print(confusion_matrix(y_test, y_pred_test))

    print(f"[Phase1-XGB] Accuracy (TEST): {acc:.4f}")

    # ---- predict for all rows and write predict_conflict ----
    df["predict_conflict"] = model.predict(X).astype(int)

    out_path = os.path.splitext(in_path)[0] + "_phase1.csv"
    df.to_csv(out_path, index=False)
    print(f"\n[Phase1-XGB] Output: {out_path}")


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"[Phase1-XGB] Execution time: {end_time - start_time:.2f} seconds")
