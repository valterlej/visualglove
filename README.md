# Visual GloVe

This repository contains an implementation of the GloVe NLP method adapted to visual features as described in the paper ''Dense Video Captioning Using Unsupervised Semantic Information'' [ArXiv link](http://).

Visual GloVe provides a dense representation encoding a co-occurrence similarity estimation with a semantic space in which short video clips with visual similar content are projected near to each other. The figure bellow shows this result.

![Examples of visual similarities. (a) Two video frag-ments with about 28 seconds from YouTube (vdBNZf90PLJ0 andvj3QSVhAhDc). They share some visual similar short clips. (b) A 2D t-SNE representation for the whole visual vocabulary. Someshared fragments are highlighted in red.](./images/visualgloveexample.png)


The code was tested on Ubuntu 20.10 with NVIDIA GPU TitanXP. Using another software/hardware might require to adapt conda environment files.

## Instalation

### Step 1: clone the repository.

```bash
git clone --recursive git@github.com:valterlej/visualglove.git
```

### Step 2: create the conda environment:

```bash
conda env create -f ./conda_env.yml
conda activate visualglove
```

### Step 3: get visual features 
a. download annotations and visual features from:

annotations (.json) https://cs.stanford.edu/people/ranjaykrishna/densevid/captions.zip

features (.npy) - https://1drv.ms/u/s!Atd3eVywQZMJgkg9BmsUa24DXgKl?e=uz9V65or

b. compute your own visual features

We suggest the git repositories:

- original: https://github.com/v-iashin/video_features or

- our fork: https://github.com/valterlej/video_features

All instructions you need are presented in them.

### Step 4: train the model with main.py script

```bash
python main.py --procedure training_and_embedding
```

This command will learn clustering and visual glove models. Additionally, it will produce the embeddings for all video stacks from *visual_features_dir* parameter.

You can extract the cluster predictions using the pre-trained cluster model for all video stacks from *visual_features_dir* with the command:

```bash
python main.py --procedure cluster_predictions
```

You can extract visual embeddings using the cluster and visual glove pre-trained models. The command is:

```bash
python main.py --procedure visual_glove_embeddings
```

Finally, you can see all parameters with:

```bash
python main.py --help
```

## Using pre-trained models in you project by importing

If you desire to incorporate our feature embedder in you project follow the steps.

