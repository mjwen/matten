# eigenn

Equivariant Invariant Graph Enabled Neural Network


## Install

This below installing guide should get you started on Mac, without using GPUs.

- [PyTorch](https://pytorch.org)
  ```bash
  conda install pytorch -c pytorch
  ```

- [PyTorch Geometrics](https://pytorch-geometric.readthedocs.io). PyG now
supports installing via conda, much easier than before. And the installing guide on
`e3nn` for PyG is not recommended.
  ```bash
  conda install pyg -c pyg -c conda-forge
  ```

- [Lightning](https://www.pytorchlightning.ai/)
  ```bash
  conda install pytorch-lightning==1.5.2 torchmetrics==0.6.0 -c conda-forge
  ```

- [e3nn](https://docs.e3nn.org/en/stable/guide/installation.html)
  ```bash
  pip install e3nn
  ```

- [nequip](https://github.com/mir-group/nequip)
  ```bash
  git clone https://github.com/mir-group/nequip.git
  cd nequip
  pip install -e .
  ```

- This repo
  ```bash
  git clone https://github.com/mjwen/eigenn.git
  cd eigenn
  pip install -e .
  ```

- [Weights & Biases](https://docs.wandb.ai/quickstart)

  We use wandb for experiments tracking and management; it is free. Get an account at
  their website and then the below commands are all you need.

  ```bash
  pip install wandb
  wandb login
  ```


## Examples

### Train on atomic property (e.g. NMR tensor)

Run this script [train_atomic.py](./scripts/train_atomic.py)
```bash
python train_atomic.py --config <config file>
```
If `config` is not provided, the default [minimal_atomic.yaml](./scripts/configs/minimal_atomic.yaml)
is used.
The model used in `train_atomic.py` is built at [atomic_tensor_model.py](./eigenn/model_factory/atomic_tensor_model.py)



### Train on structure property

Run this scirpt [train.py](./scripts/train.py)
```bash
python train.py --config <config file>
```
If `config` is not provided, the default [minimal.yaml](./scripts/configs/minimal.yaml)
is used.
The model used in `train.py` is built at [nequip_energy_model.py](./eigenn/model_factory/nequip_energy_model.py)


To get help
```bash
python train[_atomic].py --help
```

Under the hood, we use Lightning CLI to build the interface, more usage info at
[here](https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_cli.html).
