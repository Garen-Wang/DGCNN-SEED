# DGCNN
pytorch implementation of EEG Emotion Recognition Using Dynamical Graph Convolutional Neural Networks

Step 1: Download the SEED dataset, use partition.py to divide the original dataset into sessions

Step 2: Use extract_DE.py extract DE features from ExtractFeatures folder

Step 3: Run file main_DE_subject_independent.py to calculate subject-independent result, but the load path needs to be modified in advance

## 艾乐君说

前面的我都做了，直接用RUNearly.py就可以了

## 王樾说

`python RUNearly.py` 切分模型到本地

`python node.py --model_port 4001 --data_port 4011 --node_idx 1`
`python node.py --model_port 4002 --data_port 4012 --node_idx 2`
`python node.py --model_port 4003 --data_port 4013 --node_idx 3`
`python node.py --model_port 4004 --data_port 4014 --node_idx 4`

`python dispatcher.py`