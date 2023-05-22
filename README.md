# Partial Differential Equations with Deep-Learning

## Overview

This repository contains code and resources for solving partial differential equations (PDEs) using Deep Learning techniques with PyTorch. This project focuses on exploring the capabilities of PyTorch for PDE solutions and developing a deep learning-based approach for accurate and efficient PDE solving.

## Installing PyTorch Locally

PyTorch is a popular open-source deep learning library developed by Facebook's AI Research lab. This guide will walk you through installing PyTorch on your local machine.

### Step 1: Check Python Version

Before installing PyTorch, it is essential to ensure that you have a compatible version of Python installed. PyTorch supports Python versions 3.6, 3.7, 3.8, and 3.9. Open a terminal or command prompt and run the following command to check your Python version:

python --version

Make sure the output shows a compatible Python version.

### Step 2: Choose Installation Method

PyTorch can be installed using different methods, such as pip, conda, or building from source. In this guide, we will use pip for simplicity. Refer to the official PyTorch documentation for instructions if you prefer a different method.

### Step 3: Create a Virtual Environment (Optional but Recommended)

Creating a virtual environment is a good practice for isolating your Python dependencies. Open a terminal or command prompt and run the following commands to create and activate a virtual environment (replace myenv with your desired environment name):

# Create a virtual environment
python -m venv myenv
# Activate the virtual environment
# For Windows:
myenv\Scripts\activate.bat
# For macOS/Linux:
source myenv/bin/activate

### Step 4: Install PyTorch

With the virtual environment activated (if you created one), run the following command to install PyTorch using pip:

pip install torch torchvision

This command will install the latest stable version of PyTorch and its dependencies.

### Step 5: Verify the Installation

To verify that PyTorch is installed correctly, run the following Python code in a Python interpreter or Jupyter notebook:

import torch
# Check PyTorch version
print("PyTorch version:", torch.__version__)
# Create a random tensor
x = torch.randn(3, 3)
print("Random tensor:")
print(x)

If PyTorch is installed correctly, you should see the PyTorch version printed and a random tensor displayed without errors.

### Step 6: Install Additional Packages (Optional)

Depending on your project requirements, you may need to install additional packages for specific functionalities. For example, if you plan to work with GPU acceleration, you can install the necessary CUDA Toolkit and cuDNN libraries. Refer to the official PyTorch documentation for more information on additional packages.

### Conclusion

Congratulations! You have successfully installed PyTorch locally. You can now start exploring and using PyTorch for your deep-learning projects.
