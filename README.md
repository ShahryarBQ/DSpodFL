# DSpodFL
Decentralized Sporadic Federated Learning: A Unified Algorithmic Framework with Convergence Guarantees

- arXiv: [Decentralized Sporadic Federated Learning: A Unified Algorithmic Framework with Convergence Guarantees](https://arxiv.org/abs/2402.03448)

# Instruction

- **Agent.py**: Implements an edge device in the network, which conducts local gradient descent and local aggregation with its neighbors.
- **DSpodFL.py**: Main code which handles running the DSpodFL methodology, where the network graph is generated and SGD and aggregation probabilities are assigned to nodes and edges, respectively.
- **main.py**: Different experiments under various setups that are reported in the paper are obtained by running this code. Simply change the specifications in the \__init__() function and run this code.

## Citation
```
@article{zehtabi2025decentralized,
  title={Decentralized Sporadic Federated Learning: A Unified Algorithmic Framework with Convergence Guarantees},
  author={Zehtabi, Shahryar and Han, Dong-Jun and Parasnis, Rohit and Hosseinalipour, Seyyedali and Brinton, Christopher G},
  journal={International Conference on Learning Representations (ICLR)},
  year={2025}
}
```
