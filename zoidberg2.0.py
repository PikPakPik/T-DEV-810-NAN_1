import os

cwd = os.getcwd()

# Data directories, follow this structure
train_dir = os.path.join(cwd, 'chest_xray', 'train')
validation_dir = os.path.join(cwd, 'chest_xray', 'val')
test_dir = os.path.join(cwd, 'chest_xray', 'test')