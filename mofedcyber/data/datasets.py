"""Loaders for the seven intrusion detection benchmarks used in the paper.

The raw files are NOT redistributed. Download them from the official sources
(see README, "Datasets") and place the CSV files under ``data/raw/<dataset>/``.

Column names follow the public releases. If your copy uses different names,
override ``data.label_column`` / ``data.files`` in the YAML config.
"""

from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

NSL_KDD_COLUMNS = [
    "duration",
    "protocol_type",
    "service",
    "flag",
    "src_bytes",
    "dst_bytes",
    "land",
    "wrong_fragment",
    "urgent",
    "hot",
    "num_failed_logins",
    "logged_in",
    "num_compromised",
    "root_shell",
    "su_attempted",
    "num_root",
    "num_file_creations",
    "num_shells",
    "num_access_files",
    "num_outbound_cmds",
    "is_host_login",
    "is_guest_login",
    "count",
    "srv_count",
    "serror_rate",
    "srv_serror_rate",
    "rerror_rate",
    "srv_rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "srv_diff_host_rate",
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
    "label",
    "difficulty",
]

NSL_KDD_FAMILIES = {
    "normal": "Normal",
    **{
        a: "DoS"
        for a in [
            "back",
            "land",
            "neptune",
            "pod",
            "smurf",
            "teardrop",
            "apache2",
            "mailbomb",
            "processtable",
            "udpstorm",
        ]
    },
    **{a: "Probe" for a in ["ipsweep", "nmap", "portsweep", "satan", "mscan", "saint"]},
    **{
        a: "R2L"
        for a in [
            "ftp_write",
            "guess_passwd",
            "imap",
            "multihop",
            "phf",
            "spy",
            "warezclient",
            "warezmaster",
            "named",
            "sendmail",
            "snmpgetattack",
            "snmpguess",
            "worm",
            "xlock",
            "xsnoop",
            "httptunnel",
        ]
    },
    **{a: "U2R" for a in ["buffer_overflow", "loadmodule", "perl", "rootkit", "ps", "sqlattack", "xterm"]},
}


def _family_rules(rules: List[Tuple[str, str]], default: Optional[str] = None) -> Callable[[str], str]:
    """Return a mapper: first regex (case-insensitive) that matches the label wins."""
    compiled = [(re.compile(p, re.I), fam) for p, fam in rules]

    def f(label: str) -> str:
        s = str(label).strip()
        for rx, fam in compiled:
            if rx.search(s):
                return fam
        return default if default is not None else s

    return f


@dataclass
class DatasetSpec:
    name: str
    pattern: str  # glob relative to <root>/<name>/
    label_column: str
    drop_columns: List[str] = field(default_factory=list)
    header: bool = True
    column_names: Optional[List[str]] = None
    family: Optional[Callable[[str], str]] = None


REGISTRY: Dict[str, DatasetSpec] = {
    "nsl-kdd": DatasetSpec(
        "nsl-kdd",
        "KDD*.txt",
        "label",
        drop_columns=["difficulty"],
        header=False,
        column_names=NSL_KDD_COLUMNS,
        family=lambda s: NSL_KDD_FAMILIES.get(str(s).strip().rstrip("."), "Other"),
    ),
    "unsw-nb15": DatasetSpec(
        "unsw-nb15",
        "UNSW_NB15_*set.csv",
        "attack_cat",
        drop_columns=["id", "label"],
        family=_family_rules([(r"^\s*normal|^\s*$", "Normal")]),
    ),
    "cse-cic-ids2018": DatasetSpec(
        "cse-cic-ids2018",
        "*.csv",
        "Label",
        drop_columns=["Timestamp", "Flow ID", "Src IP", "Dst IP", "Src Port"],
        family=_family_rules(
            [
                (r"benign", "Benign"),
                (r"ddos|loic|hoic", "DDoS"),
                (r"dos", "DoS"),
                (r"sql", "SQL injection"),
                (r"web|xss", "Web attack"),  # before "brute": "Brute Force -Web"
                (r"brute|ftp|ssh", "Brute-Force"),
                (r"bot", "Bot"),
                (r"infil", "Infiltration"),
            ]
        ),
    ),
    "cic-iot-2023": DatasetSpec(
        "cic-iot-2023",
        "*.csv",
        "label",
        family=_family_rules(
            [
                (r"benign", "Benign"),
                (r"mirai", "Mirai"),
                (r"ddos", "DDoS"),
                (r"dos", "DoS"),
                (r"recon|vulnerabilityscan|scan", "Recon"),
                (r"spoof|mitm", "Spoofing"),
                (r"dictionary|brute", "Brute-Force"),
                (r"sql|xss|command|backdoor|upload|browser|web", "Web"),
            ]
        ),
    ),
    "edge-iiotset": DatasetSpec(
        "edge-iiotset",
        "*.csv",
        "Attack_type",
        drop_columns=[
            "Attack_label",
            "frame.time",
            "ip.src_host",
            "ip.dst_host",
            "arp.src.proto_ipv4",
            "arp.dst.proto_ipv4",
            "http.file_data",
            "http.request.full_uri",
            "icmp.transmit_timestamp",
            "http.request.uri.query",
            "tcp.options",
            "tcp.payload",
            "mqtt.msg",
            "tcp.srcport",
            "udp.port",
        ],
    ),
    "ton-iot": DatasetSpec(
        "ton-iot",
        "*.csv",
        "type",
        drop_columns=["label", "ts", "src_ip", "dst_ip"],
        family=_family_rules(
            [(r"normal", "Normal"), (r"^\s*(ddos|dos)\s*$", "DDoS"), (r"injection|xss", "Injection")]
        ),
    ),
    "5g-nidd": DatasetSpec(
        "5g-nidd",
        "*.csv",
        "Attack Type",
        drop_columns=["Label", "Attack Tool", "Unnamed: 0", "SrcAddr", "DstAddr", "StartTime"],
    ),
}


@dataclass
class TabularData:
    X: np.ndarray  # float32, standardized
    y: np.ndarray  # int64
    feature_names: List[str]
    class_names: List[str]


def _read_csvs(files: List[str], spec: DatasetSpec, max_rows: Optional[int]) -> pd.DataFrame:
    frames = []
    for f in sorted(files):
        kw = dict(low_memory=False)
        if not spec.header:
            kw.update(header=None, names=spec.column_names)
        df = pd.read_csv(f, **kw)
        df.columns = [str(c).strip() for c in df.columns]
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    return df


def _to_numeric(df: pd.DataFrame, max_categories: int = 64) -> pd.DataFrame:
    out = {}
    for c in df.columns:
        col = df[c]
        num = pd.to_numeric(col, errors="coerce")
        if num.notna().mean() > 0.95:
            out[c] = num
        else:
            n_unique = col.astype(str).nunique()
            if n_unique <= max_categories:
                out[c] = col.astype("category").cat.codes.astype(np.float64)
            # high-cardinality strings (IPs, URIs, payloads) are dropped
    X = pd.DataFrame(out)
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.loc[:, X.notna().any()]
    X = X.fillna(X.median(numeric_only=True)).fillna(0.0)
    X = X.loc[:, X.std(ddof=0) > 0]
    return X


def stratified_cap(y: np.ndarray, cap: int, rng: np.random.Generator) -> np.ndarray:
    """Indices of a class-stratified subsample of at most ``cap`` rows (keeps rare classes)."""
    if cap is None or len(y) <= cap:
        return np.arange(len(y))
    classes, counts = np.unique(y, return_counts=True)
    quota = np.maximum(1, (counts / counts.sum() * cap).astype(int))
    idx = []
    for c, q in zip(classes, quota):
        pool = np.flatnonzero(y == c)
        idx.append(rng.choice(pool, size=min(q, len(pool)), replace=False))
    return np.sort(np.concatenate(idx))


def load_dataset(
    name: str,
    root: str,
    files: Optional[List[str]] = None,
    label_column: Optional[str] = None,
    max_samples: Optional[int] = None,
    group_families: bool = True,
    seed: int = 0,
) -> TabularData:
    if name not in REGISTRY:
        raise KeyError(f"Unknown dataset '{name}'. Known: {sorted(REGISTRY)} or 'synthetic'.")
    spec = REGISTRY[name]
    if files is None:
        files = glob.glob(os.path.join(root, spec.name, spec.pattern))
    if not files:
        raise FileNotFoundError(
            f"No files for {name} under {os.path.join(root, spec.name)} (pattern {spec.pattern}). "
            "See README, section 'Datasets'."
        )
    df = _read_csvs(files, spec, max_samples)
    lab = label_column or spec.label_column
    if lab not in df.columns:
        raise KeyError(f"Label column '{lab}' not found; columns are {list(df.columns)[:20]}...")
    # CSE-CIC-IDS2018 repeats its header inside some files
    df = df[df[lab].astype(str).str.strip() != lab]
    labels = df[lab].astype(str).str.strip().fillna("Unknown")
    if group_families and spec.family is not None:
        labels = labels.map(spec.family)
    df = df.drop(columns=[c for c in spec.drop_columns + [lab] if c in df.columns])
    X = _to_numeric(df)
    classes = sorted(labels.unique(), key=lambda s: (s.lower() not in ("benign", "normal"), s))
    cmap = {c: i for i, c in enumerate(classes)}
    y = labels.map(cmap).to_numpy(np.int64)
    rng = np.random.default_rng(seed)
    keep = stratified_cap(y, max_samples, rng)
    Xv = X.to_numpy(np.float64)[keep]
    # robust standardization (log1p of magnitudes keeps heavy-tailed counters usable)
    Xv = np.sign(Xv) * np.log1p(np.abs(Xv))
    mu, sd = Xv.mean(0), Xv.std(0) + 1e-6
    Xv = ((Xv - mu) / sd).astype(np.float32)
    return TabularData(Xv, y[keep], list(X.columns), classes)


def train_test_split_stratified(data: TabularData, test_fraction: float, seed: int):
    rng = np.random.default_rng(seed)
    tr, te = [], []
    for c in np.unique(data.y):
        idx = rng.permutation(np.flatnonzero(data.y == c))
        k = max(1, int(round(len(idx) * test_fraction))) if len(idx) > 1 else 0
        te.append(idx[:k])
        tr.append(idx[k:])
    tr, te = np.concatenate(tr), np.concatenate(te)
    return tr, te
