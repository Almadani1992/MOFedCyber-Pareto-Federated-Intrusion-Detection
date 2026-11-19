# Data

Put each dataset in its own folder under `data/raw/`:

```
data/raw/nsl-kdd/          KDDTrain+.txt, KDDTest+.txt
data/raw/unsw-nb15/        UNSW_NB15_training-set.csv, UNSW_NB15_testing-set.csv
data/raw/cse-cic-ids2018/  *.csv (one file per capture day)
data/raw/cic-iot-2023/     *.csv
data/raw/edge-iiotset/     DNN-EdgeIIoT-dataset.csv
data/raw/ton-iot/          network dataset CSV(s)
data/raw/5g-nidd/          combined flow CSV
```

Official sources (check each page for its license and terms of use):

- NSL-KDD: https://www.unb.ca/cic/datasets/nsl.html
- UNSW-NB15: https://research.unsw.edu.au/projects/unsw-nb15-dataset
- CSE-CIC-IDS2018: https://www.unb.ca/cic/datasets/ids-2018.html
- CIC-IoT-2023: https://www.unb.ca/cic/datasets/iotdataset-2023.html
- Edge-IIoTset: published on IEEE DataPort by Ferrag et al. (2022)
- TON_IoT: https://research.unsw.edu.au/projects/toniot-datasets
- 5G-NIDD: published on IEEE DataPort by Samarakoon et al. (2022)

The raw files are large (CSE-CIC-IDS2018 and CIC-IoT-2023 together have more than 60M rows). Use `data.max_samples` in the config for a class-stratified subset.
