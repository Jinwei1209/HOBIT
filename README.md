# Hybrid optimization between iterative and network fine-tuning reconstructions for fast quantitative susceptibility mapping
This repo contains demo code for the paper, [Hybrid optimization between iterative and network fine-tuning reconstructions for fast quantitative susceptibility mapping](https://openreview.net/forum?id=LFaxozc7Awm).

## Dependencies
This code requires the following:

 - python 3.*
 - pytorch v1.7.1+
 - CUDA v10.1+
 
## Usage
COSMOS dataset pre-training:
```
main_COSMOS_2nets.py
```
Amortized FINE domain adaptation:
```
main_FINE_2nets_all.py
```
Hybrid optimization between iterative and network fine-tuning:
```
main_HOBIT_resnet.py
```

## Contact
To ask questions, pleasae contact jz853@cornell.edu

