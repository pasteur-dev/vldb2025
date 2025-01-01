<!-- <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/pasteur-dev/pasteur-web/master/assets/img/logo_dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/pasteur-dev/pasteur-web/master/assets/img/logo_light.svg">
    <img alt="Pasteur Logo" src="https://raw.githubusercontent.com/pasteur-dev/pasteur-web/master/assets/img/logo_light.svg" height="150px" align="left">
</picture> -->
<h1>
    Pasteur: Scaling Privacy-aware Data Synthesis<br>
    <i>Supplemental Material</i>
</h1>

This repository contains the supplemental materials for the paper 
"Pasteur: Scaling Privacy-aware Data Synthesis".
The source code of the system Pasteur is found in the main repository
[github.com/pasteur-dev/pasteur](https://github.com/pasteur-dev/pasteur).

In this repository, you will find a copy of the experiment scripts, to make
previewing them more convenient.
The original versions that should be used for reproduction and their development
history are part of the main repository (in 
[./notebooks/paper](https://github.com/pasteur-dev/pasteur/blob/v./notebooks/paper)).

# Reproduction Steps

## Prerequisites

### Python
Pasteur **requires Python 3.10+**, which you can install in ubuntu with the following
```bash
python -V # Python 3.10.9

# Ubuntu 22.04+ (has python 3.10+)
sudo apt install python3 python3-dev python3-venv
# Ubuntu 20.04 and lower
# You can use the deadsnakes PPA
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.10 python3.10-dev python3.10-venv
```

### Pasteur + Dependencies
```bash
# Clone and cd
git clone https://github.com/pasteur-dev/pasteur -b vldb
cd pasteur

# Create a virtual environment
python3.10 -m venv venv
source venv/bin/activate

# Install the frozen dependecies as they were used in the paper
pip install -r src/requirements.lock 
# Install pasteur
pip install --no-deps -e ./src
```

### Configuration
Update the raw and base location directories with where you want them.
It is required to use absolute paths for the notebooks to work due to a Kedro issue
with this version.
```bash
echo "\
base_location: $PWD/data
raw_location: $PWD/raw\
" > conf/local/globals.yml
```

### AIM and MST (optional)
Run the following commands to install AIM and MST for their experiments.
```bash
# Clone the required repositories
git clone https://github.com/dpcomp-org/hdmm external/hdmm
git -C external/hdmm checkout 7a5079a
git clone https://github.com/ryan112358/private-pgm external/private-pgm
git -C external/private-pgm checkout 6bceb36

# Add them to the venv
echo $PWD/external/hdmm/src > venv/lib/python3.10/site-packages/hdmm.pth 
echo $PWD/external/private-pgm/src > venv/lib/python3.10/site-packages/pgm.pth
echo $PWD/external/private-pgm/experiments > venv/lib/python3.10/site-packages/pgm_exp.pth

# Install their dependencies
pip install autodp==0.2 disjoint_set==0.7

# Copy slightly modified aim.py and mst.py mechanisms to private-pgm
# `command` removes `cp` aliases (ex. -i interactive) to overwrite
command cp notebooks/paper/ppgm/aim.py \
   notebooks/paper/ppgm/mst.py \
   external/private-pgm/mechanisms
```

### JAX (Optional)
One of the methods in the paper showcases marginal calculation with JAX using
both CPU and GPU modes.
To reproduce it you need JAX and (for gpu) a JAX compatible GPU.
We provide instructions for NVIDIA below.

#### Installing Nvidia drivers, CUDA and libcudnn (optional; Ubuntu only)
If you have installed nvidia drivers before and you are facing apt issues, run the following to purge the current nvidia driver packages:
```bash
# Only run me if facing issues
sudo apt purge "*nvidia*" "*cuda*"
sudo apt autoremove
sudo apt-key del 7fa2af80
```

> Warning: **NEVER** install the meta-packages `nvidia-headless-XXX`, 
> `nvidia-driver-XXX`,`nvidia-driver-XXX-server`, `nvidia-utils-XXX` directly
> (which will mark them as manually installed).
> They will break apt when nvidia releases new drivers.
> Only the `cuda` package is required to install all nvidia drivers.

JAX is not bundled with a version of cuda or cudnn currently and requires them
to be installed in the system.
JAX supports GPU only in linux environments.
Head to https://developer.nvidia.com/cuda-downloads select your distribution,
and execute the commands for `deb (network)`.


They are similar to the following, with version numbers:
```bash
# wget https://developer.download.nvidia.com/compute/cuda/repos/$distr/$arch/cuda-keyring_X.X-X_all.deb
# sudo dpkg -i cuda-keyring_X.X-X_all.deb
sudo apt-get update
sudo apt-get -y install cuda
```

Jax requires libcudnn8 and as of this writing doesn't support CUDA 12.
NVIDIA allows for multiple versions of CUDA to be installed at a time.
Install cudnn and the latest cuda version 
[listed](https://github.com/google/jax#installation) as supported by JAX.
```bash
sudo apt install libcudnn8 cuda-11-8
```

#### Installing JAX
Run one of the following commands.
```bash
# CUDA
pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
# CPU
pip install --upgrade "jax[cpu]"
```


## Experiments

### Downloading the Datasets
The paper contains Views from 2 datasets: Adult and MIMIC-IV.
We used MIMIC-IV 1.0 for the code of the paper.
Future versions of Pasteur will use the latest version.
You can download the datasets from their authors using a built-in downloader and
the following commands:
```bash
pasteur download --accept adult
pasteur download --accept mimic_iv_1_0
```
> Warning: MIMIC requires credentialed access from [physionet](https://physionet.org/content/mimiciv/2.2/).
> You will be asked for your account credentials. 
> Downloader source code: [src/pasteur/extras/download.py](https://github.com/pasteur-dev/pasteur/blob/vldb2025/src/pasteur/extras/download.py)

### Ingestion
After downloading the datasets, run the following commands to ingest them:
```bash
# The main pasteur command is "pipe", or "p"
# It is a shorter form of "kedro pipeline" that runs the pipelines pasteur has generated
# "<dataset>.ingest" runs the nodes for ingesting a dataset.
pasteur pipe adult.ingest
pasteur pipe mimic.ingest

# Likewise, "<view>.ingest" runs all the cacheable nodes of a View
# Apart from pipeline.sh, the experiments assume
# the view "ICU Charts" is ingested with its full size.
# In Pasteur, it is named "mimic_billion".
pasteur pipe mimic_billion.ingest
```

### Sidenote
Experiments ran with Jupyter have been commited with their results as used
in the paper in this folder.
They have been cleaned to remove computer specific metadata with [notebooks/utils/nbstripout](https://github.com/pasteur-dev/pasteur/blob/vldb2025/notebooks/utils/nbstripout) prior to commiting.
This script was designed to make notebooks diffable and to make two executions in
different computers appear the same by removing all changing metadata.

### Memory Usage (extended version only)
Memory use and save times based on format can be reproduced from the files 
[formats.ipynb](formats.ipynb), and [formats_save.sh](formats_save.sh).
The easiest way to compare naive to optimised encoding was to load the optimized
artifacts and de-optimise them with `df.astype()`.

Initially, `formats.ipynb` was used for the whole table.
But it was found that `pd.to_csv()` was slow and painted a bad picture.
As of this writing, `pandas` uses its own engine for saving CSV files.
When using PyArrow, CSV saves in a comparable time with Parquet files.
`formats_save.sh` uses Pasteur's export function, found in [src/pasteur/kedro/cli.py](https://github.com/pasteur-dev/pasteur/blob/vldb2025/src/pasteur/kedro/cli.py).
The underlying implementation uses PyArrow (`df.to_parquet()` uses PyArrow as well).
Time from the message `Starting export of...` was used in the paper.

### Marginal Calculations
The marginal calculations experiments (in-memory and parallelization) can be reproduced from
[marginals_inmemory.ipynb](marginals_inmemory.ipynb) and [marginals_throughput.ipynb](marginals_throughput.ipynb).

### Parallelization
The synthesis executions table was generated from [pipeline.sh](pipeline.sh).
Run the following commands to reproduce all of the table.
The results have to be extracted manually from the logs.
The system will print out how many marginals are computed during synthesis
and the combined KL measure for the ref(erence) and syn(thetic) sets.

```bash
# Assumes you created a virtual environment in venv
# Otherwise...
# export PASTEUR=pasteur

./notebooks/paper/pipeline.sh tab_adult

# Extended version only
./notebooks/paper/pipeline.sh mimic_tab_admissions

./notebooks/paper/pipeline.sh mimic_billion 1M
./notebooks/paper/pipeline.sh mimic_billion 10M
./notebooks/paper/pipeline.sh mimic_billion 100M
./notebooks/paper/pipeline.sh mimic_billion 500M
./notebooks/paper/pipeline.sh mimic_billion 1B

./notebooks/paper/pipeline.sh mimic_billion 500Msingle
```