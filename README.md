# Sparse robust linear regression with Huber's criterion in python

This code is an illustration of the use of Huber's criterion for various tasks. It consists in a toolbox provided in association with the paper:

```
Block-wise Minimization-Majorization Algorithm for Huber's Criterion: Sparse Learning and Applications, Esa Ollila and Ammar Mian
Submitted to MLSP 2020 conference.
```

It also helps with reproducibility of the results presented in the paper. It provides both matlab and python codes.

WARNING: Python version is still under debug and isn't as trustworthy as the matlab one yet but is being worked on. The results in the paper have been obtained using the matlab version.

## Files' organization

The repository is decomposed into two subdirectories:
- matlab/ which contains the matlab code. To reproduce the results presented in the paper, please run:
  - Simulation_1_Regression_example.m
  - Simulation_1_Image_denoising_example.m
- python/ which contains the python code.
  - The main functions to execute Huber regression with the MM-framework are in the package mmhuber/. 
  - Some examples of it are provided in the form of Jupyter notebooks in the subfolder notebooks/.


## Authors

The folder was created by:

* Ammar Mian, Postdoctoral researcher at Aalto University in the Department of Signal Processing and Acoustics.
  - Contact: ammar.mian@aalto.fi
  - Web: https://ammarmian.github.io/
* Esa Ollila, Professor at Aalto University in the Department of Signal Processing and Acoustics
  - Contact: esa.ollila@aalto.fi
  - Web: http://users.spa.aalto.fi/esollila/

