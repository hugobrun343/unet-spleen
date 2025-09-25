import kagglehub

# Download latest version
path = kagglehub.dataset_download("dhanvinsankaranand/spleen-segmentation-dataset")

print("Path to dataset files:", path)
