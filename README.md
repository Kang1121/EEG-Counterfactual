# EEG-Counterfactual


## Installation

1. **Clone the Repository:**  
```bash
git clone git@github.com:Kang1121/EEG-Counterfactual.git
cd EEG-Counterfactual
```

2. **Set Up the Environment:**  
```bash
conda env create --file environment.yaml
conda activate bciwinter2024
```


## Data Acquisition & Setup

1. **Download Dataset:**  
Download the required dataset from [this link](https://works.do/F7aV8dS).

2. **Place in Code Directory:**  
Ensure that the downloaded dataset is placed within the `data` directory.

## Usage Instructions

**EEG Spectrogram Generation**  
Generate EEG spectrograms using the command:
```bash
python preprocessing.py
```
**Counterfactual Explanation and Visualization**  
We provide pretrained checkpoints for direct evaluation.

For 2-Class Setting:
```bash
python explain_counterfactuals.py --config_path configs/pretrained_SAMELABELs.yaml
python visualization.py --config_path configs/pretrained_SAMELABELs.yaml
```

For 18-Class Setting:
```bash
python explain_counterfactuals.py --config_path configs/pretrained_UNIQUELABELs.yaml
python visualization.py --config_path configs/pretrained_UNIQUELABELs.yaml
```

**Train from Scratch**  
Train the model using the following command:
```bash
python trainer.py
```
