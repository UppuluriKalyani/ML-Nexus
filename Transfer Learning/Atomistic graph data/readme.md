# AtomFormer Base Model

This model is a transformer-based architecture that utilizes Gaussian pair-wise positional embeddings to train on atomistic graph data. AtomFormer is part of the AtomGen project, which supports a range of methods for pre-training and fine-tuning models on atomistic graphs.

## Model Description

AtomFormer is a transformer model designed to work with atomistic graph data. It builds on the work from Uni-Mol+ by adding pair-wise positional embeddings to the attention mask, enabling the model to leverage 3D positional information. The model has been pre-trained on a diverse set of aggregated atomistic datasets, with the target tasks being per-atom force prediction and per-system energy prediction.

In addition to the graph data, the model includes metadata about the atomic species being modeled, such as atomic radius, electronegativity, and valency. This metadata is normalized and projected into the atom embeddings within the model.

## Intended Uses & Limitations

While AtomFormer can be used for force and energy prediction, it is primarily intended to be fine-tuned for downstream tasks. The model’s performance on force and energy prediction tasks has not been extensively validated, as it was primarily used for pre-training tasks.


## Training Data

AtomFormer was trained on an aggregated S2EF dataset sourced from multiple datasets, including OC20, OC22, ODAC23, MPtrj, and SPICE. This dataset contains structures, energies, and forces for pre-training. The model was trained using formation energy, although this data is not available for OC22, as indicated by the "has_formation_energy" column in the dataset.

### Preprocessing

The model expects input in the form of tokenized atomic symbols (`input_ids`) and 3D coordinates (`coords`). During pre-training, the model also requires labels for `forces` and `formation_energy`.

The `DataCollatorForAtomModeling` utility in the AtomGen library provides dynamic padding for batching data and supports flattening the data for graph neural network (GNN)-style training.

### Pretraining Details

The model was trained on a node with 4xA40 (48 GB) GPUs for 10 epochs, which took approximately two weeks. For detailed hyperparameters and training code, refer to the [training code repository](https://github.com/VectorInstitute/AtomGen).




