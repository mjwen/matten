# eigenn

Equivariant Invariant Graph Enabled Neural Network


## Install 

This below installing guide show get you started on Mac, without using GPUs.  

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
  ```
  install pytorch-lightning
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
  

## Quick example

To train a model, run this scirpt [train.py](./scripts/train.py)
```bash
python train.py --config <config file>
```
If config is not provided, it will use the default one [minimal.yaml](./scripts/configs/minimal.yaml)

To get help
```bash
python train.py --help 
```

Under the hood, we use Lightning CLI to build the interface, more info on how to use it 
is [here](https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_cli.html). 





The actual model used `train.py` is built at [nequip_energy_model.py](./eigenn/model_factory/nequip_energy_model.py)