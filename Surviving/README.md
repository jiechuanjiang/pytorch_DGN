Update: We have fixed a small bug in surviving.py and modified some hyperparameters. The latest version is in DGN+ATOC.

## Results about DGN and baselines

| method | DGN             | Mean kernel     | IQL             | Full communication |
| ------ | --------------- | --------------- | --------------- | ------------------ |
| Reward | $-18.4 \pm 2.1$ | $-90.0 \pm 0.2$ | $-63.5 \pm 5.1$ | $-45.8 \pm 10.8$   |

Mean kernel averages the messages as the communication channel. DGN outperforms mean kernel, indicating that attention is more effective to integrate the communication messages. Full communication makes each agent communicate with all other agents, which  does not perform well, verifying the claim in ATOC that the redundant communication will negatively impact the performance.

## Results about DGN+ATOC


<img src=".ATOC.png" alt="ATOC" width="500">

ATOC could reduce the communication cost. When pruning 20% communication, the performance does not drop. When pruning 50% communication, the performance slightly drops. 
