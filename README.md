# Miniprotein stability from Sequence 

Project part of the [2021 Copenhagen Protein Biohackathon](https://biohackathon.biolib.com/event/2021-protein-edition/). 

We participated in the challenge `Predicting multi mutant miniprotein stability`

Our team name is MARS (**M**iniprotein st**A**bility f**R**om Sequence) 🌕


We built a convolutional Variational Autoencoder architecture using PyTorch to tackle the task of predicting miniprotein stability from sequence using the dataset by Rocklin et.al. Science 2017.

The idea is to learn a lower dimensional embedding of the miniprotein sequence space in the latent dimension from which the original sequence can reconstructed or from which new sequences can be sampled. Additionally, a prediction task is used to predict the stability score of the miniproteins from the learned embedding in the latent space.

This architecture has the advantage that one can sample sequences around a sequence of interest for which one knows or has predicted that it has high stability.

A sketch of the architecture is as follows:

![https://raw.githubusercontent.com/duerrsimon/mars-biohackathon/main/images/modelarchitecture.svg](Architecture)

The results for the single and multi mutant dataset are as follows:

# using a onehot encoded sequence
Each sequence is encoded using one hot encoding with 3 extra entries to denote secondary structure
- Single mutants Rp 0.72 Spearman 0.73 p<0.00001
- Multi mutants Rp 0.47 Spearman 0.35 p<0.00001

# embedding the input sequence using ProtTransBertBFD
 Single mutants Rp 0.80 Spearman 0.81 p<0.00001
[Interactive image](https://duerrsimon.github.io/mars-biohackathon/plots/embeddings_protbert_singlemutants.html)
![Single mutant prediction](plots/single.png?raw=true)

 Multi mutants Rp 0.53 Spearman 0.39 p<0.00001
[Interactive image](https://duerrsimon.github.io/mars-biohackathon/plots/embeddings_protbert_multimutants.html)
![Multi mutant prediction](plots/multi.png?raw=true)

# How to run

Create conda environment

```
conda env create -f environment.yml
```

activate environment
```
conda activate biohackathon
```

Install bioembeddings
```
pip3 install -U pip > /dev/null
pip3 install -U "bio-embeddings[all] @ git+https://github.com/sacdallago/bio_embeddings.git" > /dev/null

```

Run model
```
python embeddings_protbert_single.py 
```

Examples were run using PyTorch 1.8.1, CUDA 10.1 on a machine equipped with 2x GTX 2070, 377 GB
RAM and 2x Intel Xeon Gold 5120 CPU @ 2.20GHz
