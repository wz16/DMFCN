# DMFCN

This is the official implementation of the paper "Dilated Convolution for Time Series Learning" in ICASSP 2025. 

# dependency
The dependency is shown in requirements.txt with python 3.8

# data preparation

download the data file of TSER/UCR/UEA archives and save in the form of:

```
DMFCN/
├── dataset/
│   ├── TSER/
│       ├── AppliancesEnergy/
│           ├── AppliancesEnergy_TEST.ts
│           ├── AppliancesEnergy_TRAIN.ts
│   ├── UCRArchive_2018/
│       ├── ACSF1/
│           ├── ACSF1_TEST.tsv
│           ├── ACSF1_TRAIN.tsv
│   ├── UEAArchive_2018/
│       ├── ArticularyWordRecognition/
│           ├── ArticularyWordRecognition_TEST.ts
│           ├── ArticularyWordRecognition_TRAIN.ts
```

# run experiments

```bash
bash script/UEA.sh
bash script/UCR.sh
bash script/TSER.sh
```