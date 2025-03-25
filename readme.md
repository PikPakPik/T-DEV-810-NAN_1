# T-DEV-810 Project

## Description

This project focuses on medical image classification using chest X-rays. It includes scripts for model training and evaluation, and is organized with separate directories for training, validation, and testing images.

## Workspace Structure

```
T-DEV-810_project.pdf => Project specification
readme.md
requirements.txt => Required packages
sources.md => Contains the sources used for the project
training.py => Script for training the model
model.py => Script for running the model after training
chest_xray/ => Contains the dataset
    test/
        NORMAL/
            IM-0001-0001.jpeg
            IM-0003-0001.jpeg
            ...
        PNEUMONIA/
            ...
    train/
        NORMAL/
        PNEUMONIA/
    val/
        NORMAL/
        PNEUMONIA/
models/ => Contains saved models after training
    .gitkeep
reports/ => Contains training & testing logs and reports
    .gitkeep
    graphs/ => Contains generated graphs
        .gitkeep
```

## Team Members

- Valentin RAGOT
- Alexandre TRESSEL
- Yann PLOUHINEC
- Paul FAVREAU

## How to Run

1. Install the required packages:
```sh
pip install -r requirements.txt
```

2. Run the training script:
```sh
python training.py
```

3. Run the model after training:
```sh
python model.py
```

## Additional Information

For more details, refer to the [project specification](T-DEV-810_project.pdf).