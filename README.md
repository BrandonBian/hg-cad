## HG-CAD: Hierarchical Graph Learning for CAD Material Recommendation

### Section 1: Usage

- **Setting up the environment from scratch**
  - Python == 3.8.5
  - CUDA Toolkit ("nvcc --version") == 11.3
  - PyTorch == 1.12.0

```bash
##################################
# --- Ubuntu 20.04 LTS (X86) --- #
##################################

# Basics
sudo apt-get update
sudo apt-get upgrade -y
sudo apt-get dist-upgrade -y
sudo apt-get install build-essential

# Install Anaconda for environment control
wget https://repo.anaconda.com/archive/Anaconda3-2023.03-Linux-x86_64.sh
bash Anaconda3-*.sh
source ~/.bashrc

# Create a conda environment
conda create --name hgcad python=3.8.5
conda activate hgcad

# Install CUDA Toolkit (NVCC), version == 11.3
wget https://developer.download.nvidia.com/compute/cuda/11.3.0/local_installers/cuda_11.3.0_465.19.01_linux.run
sudo sh cuda_11.3.0_465.19.01_linux.run

# Install PyTorch

######################
# --- Windows 10 --- #
######################

# Create conda environment "hg_cad" (Python==3.8.5, CUDA==11.3, PyTorch==1.12.0)
conda env create -f environment.yml

# Activate conda environment
conda activate hg_cad

# Install torch_geometric
conda install pyg -c pyg
```

- **Obtaining the data used in paper**:
  - Make sure you have "gdown" installed (`conda install -c conda-forge gdown`)
  - Unzip (`unzip <file>.zip`) and place inside a folder (e.g., `./dataset/`)
  - **If "zipfile corrupt" error**: try `zip -FF Corrupted.zip --out New.zip` (PS: install zip with `sudo apt-get install zip`)
```bash
# Full dataset (on paper)
gdown https://drive.google.com/uc?id=1f9jIgzSHRuT3jPMO17Vl0EYvq4v5kTue

# Top 40% dataset (much smaller, suitable for sanity checks and test developments)
gdown https://drive.google.com/uc?id=1h4iwI8tuOicZhjHGr60mgRu53kUDwEEM

# Obtain the train and val spit - TODO
```

- **Training**: Training on train data, and automatically performs testing on test data when training is finished.
```bash
# Additional tuning knobs included INSIDE python file -- see classification.py
python classification.py train --dataset_path [path/to/dataset] --max_epoch 100 --batch_size 16 --gpus 1
```

- **Testing**: Testing on test data based on checkpoint and random seed obtained from previously finished training.
```bash
python classification.py test --dataset_path [path/to/dataset] --checkpoint [path/to/best.ckpt] --random_seed [seed integer]
```

- **Inference**: on a *single* sample assembly, and produce material label predictions for *all* of its bodies.
```
# Additional tuning knobs included INSIDE python file -- see inference.py
python inference.py single_sample --inference_sample [path/to/inference/sample] --checkpoint [path/to/best.ckpt] --vocab [path/to/vocab.pickle]
```

- **Inference**: on *multiple* sample assemblies, and produce material label predictions for *one random* body per assembly.
```
python inference.py multiple_sample --inference_sample [path/to/inference/samples] --checkpoint [path/to/best.ckpt] --vocab [path/to/vocab.pickle]
```

---

### Section 2: Data Processing Tools for HG-CAD

| Tool | Description |
| -- | -- |


---

### Section 3: Baselines

| Model | Data Format | Repository | Data Processing Tools | Results |
| -- | -- | -- | -- | -- |
| PointNet [*](https://doi.org/10.48550/arXiv.1612.00593) | PointCloud OFF Files | [Adapted Version](https://github.com/BrandonBian/pointnet-tensorflow) | [OBJ to OFF](TODO) | [Link](https://drive.google.com/drive/folders/1gD5NRNyzzHVVn0mfhRETty_dEB-Wut0N?usp=share_link) |
| UV-Net [*](https://doi.org/10.48550/arXiv.2006.10211) | DGL Graph BIN Files | [Adapted Version](https://github.com/BrandonBian/UV-Net) | [STEP to BIN](TODO); [Visualization](TODO) | [Link](https://drive.google.com/drive/folders/1GS14bYIzT5ut42Tr50X6nOTdyKpmDXdQ?usp=share_link) |
| Human Baseline | OBJ and PNG files | [Original Version](https://github.com/BrandonBian/Human-Baseline) | [Prompt Generation](TODO) | - |
