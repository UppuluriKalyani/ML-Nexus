# High-fidelity Mesh Generation with Interactive Refinement

## Overview

This model is a high-fidelity, two-stage 3D mesh generation system designed to deliver smooth and accurate geometries while allowing interactive refinement of details. It starts with generating a basic, coarse mesh using a 3D diffusion model, which is then refined using multi-view normal maps. This interactive approach enables adjustments similar to digital sculpting tools, allowing users to produce and fine-tune realistic 3D objects efficiently.

## Importance

Traditional 3D generation models often face limitations with extended processing times, irregular mesh topologies, and difficulty handling user edits. This model addresses these issues by:
- Providing efficient, coarse mesh generation with regular topologies in seconds.
- Refining meshes interactively, allowing artists to produce high-fidelity results with flexibility.
- Reducing noise and inconsistencies in generated meshes, making it easier for users to adapt the meshes to their projects.

These improvements make the model suitable for applications in gaming, animation, and 3D content creation, where precision and flexibility are crucial.

![diagram](https://github.com/user-attachments/assets/55972f9a-0762-4879-9cc3-15acd3f48914)

## Steps to Use the Model

### 1. **Environment Setup**

Set up the environment with the necessary dependencies. An environment file is provided for installation.

### 2. **Model Inference**

The model can generate 3D meshes based on an input image or text prompt. Here’s how to use it:
- **Single-view input**: Run `inference.py` with a single image or text prompt for basic 3D generation.
- **Multi-view input**: Specify images from different perspectives for more accurate results.
- **Interactive Refinement**: Use the interactive mode to adjust surface details and improve mesh fidelity.

### 3. **Refinement Options**

Leverage the normal-based geometry refinement tool to enhance surface details. The tool can operate in an automatic mode or allow for interactive refinement by the user, making it possible to achieve highly realistic results.

### 4. **Training from Scratch**

The repository includes training code for custom dataset preparation. Instructions are available in the `configs` folder, including sample configurations.

### 5. **Gradio Demo**

For a quick visual demonstration, use the Gradio interface to test the model on your local machine. Run `gradio_app.py` with the model path to initiate the demo.

## Pretrained Models and Resources

Pretrained weights are provided for immediate use and can be downloaded from a model repository. Simply download and place the models in the specified directory for seamless integration.

### Future Updates

The model will continue to be improved with additional pretrained checkpoints and further refinements to increase efficiency and expand the range of compatible input data. 

