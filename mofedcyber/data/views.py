"""Split flow features into the three views used by the CAT detector (Section IV-A).

protocol (p): ports, flags, service and protocol indicators
temporal (t): duration, inter-arrival statistics, rates, bursts, active/idle times
payload  (s): byte ratios, entropy, packet-size moments and everything else
"""

from __future__ import annotations

import re
from typing import Dict, List

PROTOCOL_PATTERNS = [
    r"port",
    r"proto",
    r"flag",
    r"service",
    r"\bsyn\b|syn_|_syn|syn\s",
    r"fin",
    r"rst",
    r"psh",
    r"ack",
    r"urg",
    r"ece",
    r"cwr|cwe",
    r"state",
    r"ttl",
    r"win",
    r"land",
    r"tcp\.",
    r"udp\.",
    r"icmp",
    r"arp",
    r"dns",
    r"http\.(request\.)?method",
    r"mqtt\.(proto|conflag|hdrflag|msgtype)",
    r"is_",
    r"logged_in",
    r"conn_state",
    r"modbus",
    r"^(sttl|dttl|swin|dwin|smeansz)$",
]
TEMPORAL_PATTERNS = [
    r"dur",
    r"iat",
    r"time",
    r"rate",
    r"/s\b|_s$|per_s|ps$",
    r"active",
    r"idle",
    r"jit",
    r"load",
    r"intpkt|sintpkt|dintpkt",
    r"tcprtt|synack|ackdat",
    r"count",
    r"burst",
    r"bulk",
    r"interval",
    r"ts_",
    r"lifetime",
]


def assign_views(feature_names: List[str]) -> Dict[str, List[int]]:
    views = {"protocol": [], "temporal": [], "payload": []}
    for i, name in enumerate(feature_names):
        n = name.lower()
        if any(re.search(p, n) for p in PROTOCOL_PATTERNS):
            views["protocol"].append(i)
        elif any(re.search(p, n) for p in TEMPORAL_PATTERNS):
            views["temporal"].append(i)
        else:
            views["payload"].append(i)
    # every view must be non-empty: borrow from the largest one if needed
    for v in views:
        if not views[v]:
            donor = max(views, key=lambda k: len(views[k]))
            views[v].append(views[donor].pop())
    return views


def even_views(num_features: int) -> Dict[str, List[int]]:
    """Fallback for anonymous features (e.g. synthetic data): split into three equal blocks."""
    idx = list(range(num_features))
    k = num_features // 3
    return {"protocol": idx[:k], "temporal": idx[k : 2 * k], "payload": idx[2 * k :]}
