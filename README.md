# FeedBack Reconstruction

Codebase for Reconstructing feedback representations in ventral visual pathway with a generative adversarial
autoencoder. In this repository, we only have PyTorch implementation. Please follow the steps below to ensure that the
code work correctly.

# Install Python requirements

Before running the scripts, install all the python libraries:

```
pip install -r requirements.txt
```

Additionally, we pre-installed `cuda/10.1`, `cudnn/7.6.5`, and `nccl/2.5.6`. Instructions can be found
here https://developer.nvidia.com/cuda-toolkit-archive

# Training

There is one main script that perform most operations used in the paper: `main.py`

To start training with default options, this command will do:

```
python main.py train_aae
```

The command above will train an adversarial autoencoder. To change that, you can replace the function with:

```
python main.py train_vae
```

This will train a variational autoencoder.

# Testing

To get the RDMs after training you'll need to run:

 ```
python main.py gen_rdm
```

Make sure that you have the saved model in the `./model` directory.

# Saved Models

Model for the main result can be found in the in [`models`](https://github.com/haideraltahan/FeedBack-Reconstruction/tree/main/models/aae_dprior) folder.

Please make sure to maintain the folder structure.
