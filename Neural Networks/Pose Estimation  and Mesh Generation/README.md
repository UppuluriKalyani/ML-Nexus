# Pose Estimation and Mesh Generation for UP-3D Dataset

This project implements pose estimation and mesh generation using the **UP-3D dataset**. It involves preprocessing the dataset, training PosePrior and ShapePrior models, evaluating their performance, and rendering 3D meshes with Open3D.

## UP-3D Dataset

The **UP-3D (Unite the People)** dataset provides images and corresponding 3D mesh ground truths of human poses. It includes annotations such as 2D keypoints, 3D joint positions, body part segmentation, and 3D surface meshes, making it suitable for tasks like pose estimation, shape modeling, and mesh generation.

## Project Workflow

### 1. Preprocessing

Preprocessing steps are carried out based on the UP-3D dataset's requirements, which may include:

- Normalizing 2D keypoints.
- Scaling 3D mesh coordinates.
- Data augmentation (e.g., flipping, rotation).
- Splitting data into training, validation, and test sets.

### 2. Model Training

The project utilizes two main models for training:

- **PosePrior Model**: Predicts human pose from the given inputs (e.g., 2D keypoints).
- **ShapePrior Model**: Predicts human body shape based on pose information.

### 3. Evaluation

The provided code evaluates the trained models on the test dataset. Evaluation metrics, such as loss, are used to assess the model's performance.

### 4. Mesh Rendering

Rendered 3D meshes are visualized using **Open3D**. After training and evaluating the models, the mesh generation output is rendered for better visualization of the results.

## Future Improvements

- Fine-tuning model architectures for better performance.
- Incorporating additional data augmentation techniques.
- Experimenting with different mesh rendering techniques.

## Acknowledgments

- **UP-3D Dataset**: For providing detailed annotations for human pose and shape modeling.
- **Open3D**: For supporting 3D visualization and mesh rendering.

## License

This project is licensed under the [Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)](https://creativecommons.org/licenses/by-sa/4.0/) license.

We make the code as well as the datasets available for academic or non-commercial use under the Creative Commons Attribution-Noncommercial 4.0 International license.

Contact us by E-Mail.
Hosted by the Perceiving Systems Department, part of the Max Planck Institute for Intelligent Systems.

Based on a template by Tyler Fowle.

### Attribution

When using or sharing this project, please include the following attribution:

> "This work is based on [Unite the People: Closing the Loop Between 3D and 2D Human Representations] by [Lassner, Christoph and Romero, Javier and Kiefel, Martin and Bogo, Federica and Black, Michael J. and Gehler, Peter V.], licensed under CC BY-SA 4.0."

### Disclaimer

The project is provided "as is," without warranty of any kind, express or implied. The CC license may not cover all permissions necessary for your intended use. Additional rights such as publicity, privacy, or moral rights may apply.
