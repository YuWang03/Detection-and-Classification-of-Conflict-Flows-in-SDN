#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import pandas as pd
import ipaddress
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

from xgboost import XGBClassifier


# -----------------------------
# IP relation feature (single feature)
# -----------------------------
def _parse_ip_any(s: str):
    """Return (is_network, ip_obj_or_net, prefixlen_or_32)."""
    s = str(s).strip()
    if s in ("0", "", "None") or s.lower() == "nan":
        return (False, None, 32)

    try:
        if "/" in s:
            net = ipaddress.ip_network(s, strict=False)
            return (True, net, int(net.prefixlen))
        else:
            ip = ipaddress.ip_address(s)
            return (False, ip, 32)
    except Exception:
        return (False, None, 32)


def build_ip_relation(src_ip: str, dst_ip: str) -> str:
    """
    Build ONE categorical feature representing "IP address relation"
    using info available in CSV:
      - dst is host vs network
      - dst prefixlen (if network)
      - whether src/dst share same /24
    """
    src_s = str(src_ip).strip()
    dst_s = str(dst_ip).strip()

    dst_is_net, _, dst_plen = _parse_ip_any(dst_s)

    def _first3(ip_s: str):
        ip_s = ip_s.split("/")[0].strip()
        parts = ip_s.split(".")
        return ".".join(parts[:3]) if len(parts) >= 3 else ""

    same24 = 1 if (_first3(src_s) and _first3(src_s) == _first3(dst_s)) else 0

    if dst_is_net:
        return f"dst_net_p{dst_plen}_same24_{same24}"
    else:
        return f"dst_host_same24_{same24}"


def _normalize_type(s: str) -> str:
    s = str(s).strip()
    if s == "" or s.lower() == "nan":
        return "normal"
    return s.lower()


def main():
    t0 = time.time()

    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--test_size", type=float, default=0.3)
    ap.add_argument("--random_state", type=int, default=42)

    # XGBoost hyperparams
    ap.add_argument("--n_estimators", type=int, default=300)
    ap.add_argument("--max_depth", type=int, default=6)
    ap.add_argument("--learning_rate", type=float, default=0.1)
    ap.add_argument("--subsample", type=float, default=0.9)
    ap.add_argument("--colsample_bytree", type=float, default=0.9)
    ap.add_argument("--reg_lambda", type=float, default=1.0)
    args = ap.parse_args()

    if not os.path.exists(args.input):
        print(f"[ERROR] Input file not found: {args.input}")
        sys.exit(1)

    df = pd.read_csv(args.input)

    required_cols = ["predict_conflict", "priority", "src_ip", "dst_ip", "action", "conflict_type"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"[ERROR] Missing required columns in CSV: {missing}")
        print(f"        Existing columns: {list(df.columns)}")
        sys.exit(1)

    # Normalize labels
    df["conflict_type"] = df["conflict_type"].apply(_normalize_type)

    # 7 conflict types only
    allowed_types = [
        "redundancy",
        "shadowing",
        "generalization",
        "correlationa",
        "correlationb",
        "overlap",
        "imbrication",
    ]

    # -----------------------------
    # Feature engineering
    # -----------------------------
    df["priority"] = pd.to_numeric(df["priority"], errors="coerce").fillna(0).astype(int)
    df["action"] = df["action"].fillna("unknown").astype(str)
    df["ip_relation"] = [build_ip_relation(s, d) for s, d in zip(df["src_ip"], df["dst_ip"])]

    # -----------------------------
    # df_eval (你要評估的集合)
    # predict_conflict==1 且 真實 conflict_type != normal
    # -----------------------------
    df_eval = df[
        (df["predict_conflict"].astype(int) == 1) &
        (df["conflict_type"] != "normal")
    ].copy()

    if len(df_eval) == 0:
        print("[WARN] df_eval is empty: no rows where predict_conflict==1 and conflict_type!=normal")
        sys.exit(0)

    # -----------------------------
    # Train set: true conflicts only (7 types) from FULL df
    # -----------------------------
    df_true_conf = df[df["conflict_type"].isin(allowed_types)].copy()
    if len(df_true_conf) < 10 or df_true_conf["conflict_type"].nunique() < 2:
        print("[ERROR] Not enough true-conflict samples (or not enough classes) to train Phase2 XGB.")
        sys.exit(1)

    X_tc = df_true_conf[["priority", "ip_relation", "action"]]
    y_tc_str = df_true_conf["conflict_type"]

    # Label encode y into 0..6
    le = LabelEncoder()
    le.fit(allowed_types)  # 固定 7 類的順序
    y_tc = le.transform(y_tc_str)

    # Preprocess X
    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["ip_relation", "action"]),
            ("num", "passthrough", ["priority"]),
        ]
    )

    xgb = XGBClassifier(
        objective="multi:softprob",
        eval_metric="mlogloss",
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        reg_lambda=args.reg_lambda,
        random_state=args.random_state,
        n_jobs=-1
    )

    pipe = Pipeline([("pre", pre), ("clf", xgb)])

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_tc, y_tc,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y_tc
    )

    pipe.fit(X_tr, y_tr)

    # Holdout sanity (true conflicts only)
    y_pred_te = pipe.predict(X_te)
    acc_holdout = accuracy_score(y_te, y_pred_te)

    print("\n[Phase2] XGBoost trained on TRUE conflicts only (7 classes)")
    print(f"  True-conflict pool: {len(df_true_conf)}")
    print(f"  Train: {len(X_tr)} | Test: {len(X_te)}")
    print(f"  Holdout Accuracy (TRUE conflicts only): {acc_holdout:.4f}")

    # -----------------------------
    # Predict & evaluate ONLY df_eval
    # -----------------------------
    X_eval = df_eval[["priority", "ip_relation", "action"]]
    y_true_str = df_eval["conflict_type"].tolist()
    y_true = le.transform(y_true_str)  # 轉成 0..6

    y_pred = pipe.predict(X_eval)

    acc = accuracy_score(y_true, y_pred)

    # 回轉成字串做 report
    y_true_lbl = le.inverse_transform(y_true)
    y_pred_lbl = le.inverse_transform(y_pred)

    rep = classification_report(
        y_true_lbl, y_pred_lbl,
        labels=allowed_types,
        zero_division=0,
        output_dict=True
    )
    macro_p = rep["macro avg"]["precision"]
    macro_r = rep["macro avg"]["recall"]
    macro_f1 = rep["macro avg"]["f1-score"]

    print("\n[Phase2 Evaluation on df_eval ONLY]")
    print(f"  df_eval size       : {len(df_eval)}")
    print(f"  Accuracy           : {acc:.4f}")
    print(f"  Macro Precision    : {macro_p:.4f}")
    print(f"  Macro Recall       : {macro_r:.4f}")
    print(f"  Macro F1           : {macro_f1:.4f}")

    print("\n[Classification Report on df_eval ONLY]")
    print(classification_report(y_true_lbl, y_pred_lbl, labels=allowed_types, zero_division=0))

    t1 = time.time()
    print(f"\n[Phase2] Execution time: {t1 - t0:.4f} sec")


if __name__ == "__main__":
    main()
