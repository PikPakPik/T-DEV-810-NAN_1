# T-DEV-810 Project

## Description

This project focuses on medical image classification using chest X-rays. It includes scripts for model training and evaluation, and is organized with separate directories for training, validation, and testing images.

__Documentation__ : [https://epitechfr-my.sharepoint.com/:w:/g/personal/alexandre_tressel_epitech_eu/EThtgguGLnRHqttzBHMiCf8BpL75bbGoehW6-W6AgGIPCQ](https://epitechfr-my.sharepoint.com/:w:/g/personal/alexandre_tressel_epitech_eu/EThtgguGLnRHqttzBHMiCf8BpL75bbGoehW6-W6AgGIPCQ)

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

## Additional Information

For more details, refer to the [project specification](T-DEV-810_project.pdf).

## Sources

The sources used for the project are available in the file [sources.md](sources.md).

## How to use

### To use the CNN model, follow these steps:
1. Open cnn_training.ipynb in Jupyter Notebook.
2. Run the cells in order to train the model.
3. After training, the model will be saved in the models directory.
4. To test, open cnn_predict.ipynb in Jupyter Notebook and run the cells.

### To use the ResNet model, follow these steps:
1. Open ResNet.ipynb in Jupyter Notebook.
2. Run the cells in order to train the model.

### To use the VGG model, follow these steps:
1. Open vgg16.ipynb in Jupyter Notebook.
2. Run the cells in order to train the model.

