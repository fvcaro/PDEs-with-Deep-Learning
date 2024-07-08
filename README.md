# Partial Differential Equations with Deep-Learning

## Overview

This repository contains code and resources for solving partial differential equations (PDEs) using Deep Learning techniques with PyTorch. The project focuses on exploring PyTorch's capabilities for PDE solutions and developing a deep learning-based approach for accurate and efficient PDE solving.

We aim to use Physics-Informed Neural Networks (PINNs) to enhance Deep Learning results in scenarios where traditional numerical methods encounter limitations. We particularly focus on applications in numerical relativity, where our approach could lead to significant advancements.

## Installing PyTorch Locally

Based on the Torch library, PyTorch originated at Meta AI and now operates under the Linux Foundation. This machine-learning library excels in computer vision and natural language processing applications. As one of the two most popular machine learning libraries alongside TensorFlow, PyTorch is free and open-source software under the modified BSD license. Although development primarily focuses on the highly polished Python interface, PyTorch also includes a C++ interface. This guide will walk you through installing PyTorch on your local machine.

## Step 1: Check Python Version

Before installing PyTorch, ensure you have a compatible version of Python installed. Python 3.8 - 3.11 is recommended. Open a terminal or command prompt and run the following command to check your Python version:

```sh
python --version
```
Make sure the output shows a compatible Python version. You can install Python through the Anaconda package manager, Homebrew, or the Python website. 

## Step 2: Choose Installation Method

PyTorch can be installed using different methods, such as pip, conda, or building from source. In this guide, we will use conda for simplicity. Refer to the official PyTorch documentation for instructions if you prefer a different method.

## Step 3: Create and Activate a Virtual Environment (Optional but Recommended)

Creating a virtual environment is a good practice for isolating your Python dependencies. Follow these steps to create and activate a virtual environment:

1. **Open a terminal or command prompt**.

2. **Create a virtual environment** (replace `your_env_name` with your desired environment name):
    ```sh
    conda create --name your_env_name
    ```

3. **Activate the virtual environment**:
    ```sh
    conda activate your_env_name
    ```

## Step 4: Install PyTorch

Let's install PyTorch with a version compatible with CUDA 12.1, which is available in the default channels. Run the following command:

```sh
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

## Step 5: Verify the Installation

To verify that PyTorch is installed correctly, run the following Python code in a Python interpreter or Jupyter notebook:

```sh
import torch
Check PyTorch version
print("PyTorch version:", torch.__version__)
Create a random tensor
x = torch.randn(3, 3)
print("Random tensor:")
print(x)
```

If PyTorch is installed correctly, you should see the PyTorch version printed and a random tensor displayed without errors.

## Step 6: Install Additional Packages (Optional)

Depending on your project requirements, you may need to install additional packages for specific functionalities. For example, if you plan to work with GPU acceleration, you can install the necessary CUDA Toolkit and cuDNN libraries. Refer to the official PyTorch documentation for more information on additional packages.

## Conclusion

Congratulations! You have successfully installed PyTorch locally. You can now start exploring and using PyTorch for your deep-learning projects.