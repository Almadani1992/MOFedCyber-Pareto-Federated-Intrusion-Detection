"""All values reported in the MOFedCyber results report (means over 30 seeds).
Nothing here is simulated or interpolated; every number is a reported value."""

DATASETS = ["NSL-KDD", "UNSW-NB15", "CIC-IDS2018", "CIC-IoT2023", "Edge-IIoTset", "TON_IoT", "5G-NIDD"]

# Table: macro-F1 (%) per dataset, mean, average rank
F1 = {
    "FedAvg": ([88.2, 81.7, 92.3, 89.5, 90.8, 87.6, 88.9], 88.4, 11.0, "Single-objective"),
    "FedProx": ([89.1, 83.2, 93.1, 90.7, 91.5, 88.4, 89.6], 89.4, 9.0, "Single-objective"),
    "FedNova": ([89.6, 83.9, 93.4, 91.2, 91.9, 89.0, 90.1], 89.9, 7.1, "Single-objective"),
    "SCAFFOLD": ([91.3, 86.5, 94.8, 92.6, 93.2, 91.1, 92.0], 91.6, 5.0, "Single-objective"),
    "FedMGDA+": ([93.7, 89.4, 95.6, 93.8, 94.5, 92.7, 93.4], 93.3, 4.0, "Multi-objective"),
    "MOO-FL-εC": ([94.1, 90.1, 95.9, 94.2, 94.8, 93.2, 93.8], 93.7, 3.0, "Multi-objective"),
    "DP-FedAvg": ([86.5, 79.8, 90.6, 87.9, 89.1, 85.9, 87.3], 86.7, 12.0, "Privacy-preserving"),
    "LDP-Fed": ([84.2, 77.3, 88.9, 85.8, 87.2, 83.6, 85.0], 84.6, 13.0, "Privacy-preserving"),
    "Krum": ([89.8, 83.7, 93.2, 90.9, 91.7, 88.6, 89.8], 89.7, 7.9, "Byzantine-robust"),
    "Median": ([88.7, 82.1, 92.5, 89.8, 90.9, 87.8, 89.0], 88.7, 10.0, "Byzantine-robust"),
    "Bulyan": ([90.5, 85.2, 94.0, 91.7, 92.4, 89.7, 90.6], 90.6, 6.0, "Byzantine-robust"),
    "FLAME": ([95.2, 91.3, 96.4, 95.0, 95.7, 94.1, 94.5], 94.6, 1.9, "Byzantine-robust"),
    "MOFedCyber": ([97.8, 94.2, 98.1, 97.5, 97.9, 96.6, 94.2], 96.6, 1.1, "Proposed"),
}

DATASET_INFO = [  # name, samples, features, attack classes, domain
    ("NSL-KDD", 148_517, 41, 4, "Enterprise (legacy)"),
    ("UNSW-NB15", 2_540_044, 49, 9, "Enterprise"),
    ("CSE-CIC-IDS2018", 16_232_943, 80, 14, "Enterprise"),
    ("CIC-IoT-2023", 46_686_579, 47, 33, "Consumer IoT"),
    ("Edge-IIoTset", 2_219_201, 61, 14, "Industrial IoT"),
    ("TON_IoT", 22_339_021, 42, 7, "Mixed IoT"),
    ("5G-NIDD", 1_215_890, 52, 8, "5G edge"),
]

PARETO = {  # HV, IGD, spacing, non-dominated count
    "FedMGDA+": (0.612, 0.085, 0.094, None),
    "MOO-FL-εC": (0.671, 0.071, 0.083, 21),
    "MOFedCyber": (0.834, 0.049, 0.067, 45),
}

CONV = {  # rounds to reach own plateau / target, cumulative GB at convergence
    "MOFedCyber": dict(rounds=94, gb=62.8, reaches_target=True),
    "FLAME": dict(rounds=147, gb=None, reaches_target=False),
    "FedAvg": dict(rounds=172, gb=198.1, reaches_target=False),
}
COMM_PER_ROUND = dict(measured=42.0, predicted=42.5)

EPS = [0.1, 0.25, 0.5, 1, 2, 5, 10]
PRIV_F1 = {
    "MOFedCyber": [91.7, 94.0, 95.4, 96.4, 97.1, 97.6, 97.8],
    "DP-FedAvg": [78.4, 82.6, 85.1, 87.1, 89.0, 90.7, 91.5],
    "LDP-Fed": [72.1, 77.0, 80.6, 83.5, 86.0, 88.0, 89.0],
}
PRIV_NODP = {"MOFedCyber": 98.1, "DP-FedAvg": 92.3, "LDP-Fed": 90.1}
MIA_EPS = ["No DP", "ε = 10", "ε = 1", "ε = 0.1"]
MIA = {"DP-FedAvg": [78.3, 62.4, 54.1, 50.8], "MOFedCyber": [71.2, 55.7, 51.3, 50.2]}
SIGMA_Z = [0, 2.41, 17.3, 144]

BYZ_X = [0, 5, 10, 15, 20, 25, 30, 35, 40]
BYZ = {
    "MOFedCyber": [98.1, 97.9, 97.5, 97.0, 96.3, 95.4, 94.2, 90.1, 84.5],
    "FLAME": [96.4, 96.0, 95.4, 94.5, 93.4, 91.8, 89.6, 84.2, 77.0],
    "Bulyan": [94.0, 93.3, 92.4, 91.1, 89.4, 87.0, 83.5, 76.4, 66.0],
    "Krum": [93.2, 92.4, 91.3, 89.7, 87.5, 84.4, 79.8, 71.0, 60.5],
    "Median": [92.5, 91.6, 90.3, 88.5, 86.0, 82.5, 77.4, 68.5, 57.0],
    "FedAvg": [92.3, 86.5, 79.4, 71.2, 62.5, 53.8, 45.2, 37.0, 30.1],
}

DROP_X = [0, 10, 20, 30, 40, 50]
DROP = {
    "MOFedCyber": ([98.1, 97.8, 97.3, 96.7, 96.0, 94.6], [0.20, 0.21, 0.22, 0.24, 0.26, 0.30]),
    "FLAME": ([96.4, 95.6, 94.5, 92.9, 90.8, 87.5], [0.25, 0.27, 0.31, 0.35, 0.42, 0.51]),
    "SCAFFOLD": ([94.8, 93.9, 92.6, 90.8, 88.4, 85.0], [0.30, 0.32, 0.36, 0.42, 0.49, 0.59]),
    "FedAvg": ([92.3, 90.9, 89.0, 86.4, 83.0, 78.4], [0.37, 0.41, 0.47, 0.55, 0.65, 0.79]),
}

SCAL_N = [10, 25, 50, 100, 200, 300, 400, 500]
SCAL = {
    "MOFedCyber": [2.1, 2.6, 3.5, 5.3, 9.1, 12.8, 16.6, 20.5],
    "FLAME": [2.0, 2.4, 3.2, 4.8, 8.2, 11.6, 15.1, 18.5],
    "FedAvg": [1.8, 2.1, 2.6, 3.6, 5.7, 7.8, 9.9, 12.0],
}
NSGA_SHARE = [3.1, 3.6, 4.4, 5.2, 6.3, 7.0, 7.6, 8.0]

ENERGY = {  # computation, communication, idle (kJ per round)
    "FedAvg": (4.2, 6.8, 1.5),
    "FedProx": (4.5, 6.9, 1.5),
    "SCAFFOLD": (5.0, 6.9, 1.5),
    "FLAME": (4.4, 6.4, 1.3),
    "MOFedCyber": (4.4, 2.6, 1.0),
}

ALPHA = [0.05, 0.1, 0.3, 0.5, 1, 5]
NONIID = {
    "MOFedCyber": [94.9, 96.4, 97.6, 98.1, 98.4, 98.7],
    "FLAME": [89.9, 93.0, 95.4, 96.4, 97.0, 97.5],
    "FedMGDA+": [87.6, 91.2, 94.0, 95.6, 96.3, 96.9],
    "SCAFFOLD": [84.2, 88.6, 92.4, 94.8, 95.6, 96.3],
    "FedAvg": [76.8, 83.2, 89.4, 92.3, 93.6, 94.8],
}

HEADS = [  # head, specialty, score, drop pp
    ("H1", "Protocol–port pairs", 0.81, 3.2),
    ("H2", "Flow duration", 0.73, 2.4),
    ("H3", "Packet-size statistics", 0.78, 2.8),
    ("H4", "TCP flags", 0.69, 2.1),
    ("H5", "Inter-arrival times", 0.84, 3.5),
    ("H6", "Byte ratios", 0.71, 2.3),
    ("H7", "Burst patterns", 0.86, 3.7),
    ("H8", "Encoding signatures", 0.74, 2.5),
]

ABLATION = [  # variant, F1, delta
    ("Full MOFedCyber", 96.6, 0.0),
    ("w/o adaptive operating point", 95.9, -0.7),
    ("w/o cross-attention", 94.8, -1.8),
    ("w/o APGD (uniform weights)", 93.1, -3.5),
    ("w/o NSGA-III (random archive)", 92.7, -3.9),
    ("w/o Bulyan (mean aggregation)", 88.2, -8.4),
]
NO_DP = 97.4
