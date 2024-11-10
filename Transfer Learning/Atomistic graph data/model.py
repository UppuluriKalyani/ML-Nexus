import kagglehub

# Download latest version
path = kagglehub.model_download("tedlord/atomformer/pyTorch/default")

print("Path to model files:", path)