# The instruction for the Machine Learning (1000 for example)
## 將 instruction 中的 1000 改成其他數字即可


### For the Decision Tree
#### For the Detection
python3 ./phase1_detect_dt.py dataset/1000.csv

#### For the Classification
python3 phase2_classify_dt.py --input dataset/1000_phase1.csv

### For the XGBoost
#### For the Detection
python3 phase1_detect_XGB.py dataset/1000.csv

#### For the Classification
python3 phase2_classify_XGB.py --input dataset/1000_phase1.csv
