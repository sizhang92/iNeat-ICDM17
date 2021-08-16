# Incomplete-Network-Alignment
 Matlab code for iNeat proposed in our paper "iNEAT: Incomplete Network Alignment" in ICDM17

## Overview
The package contains the following files:
- iNeat: the main algorithm file of the proposed iNeat algorithm
- greedy_match.m: the greedy matching algorithm as the post-processing to obtain the one-to-one mapping
- run_ineat.m: a demo code file of the proposed iNeat algorithm
- run_finalP_incomplete.m: a demo code file of the baseline method FINAL-P
- run_isorank_incomplete.m: a demo code file of the baseline method IsoRank
- run_fgm_incomplete.m: a demo code file of the baseline method RRWM
- Gordian-v2-(1%-25%).mat: the datasets contain the adjacency matrices with (1%-25%) missing edges

## Usage
Please refer to the demo code file demo.m and the descriptions in each file for the detailed information. 
The code can be only used for academic purpose and please kindly cite our published paper.

## Reference
Zhang, Si, Hanghang Tong, Jie Tang, Jiejun Xu, and Wei Fan. "iNEAT: Incomplete Network Alignment." In Data Mining (ICDM), 2017 IEEE International Conference on, pp. 1189-1194. IEEE, 2017.
