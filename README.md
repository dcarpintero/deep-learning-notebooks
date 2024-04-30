# Practical Deep Learning 

Learning Projects to Get Started with Deep Learning

[![GitHub license](https://img.shields.io/github/license/dcarpintero/deep-learning-notebooks.svg)](https://github.com/dcarpintero/deep-learning-notebooks/blob/master/LICENSE)
[![GitHub contributors](https://img.shields.io/github/contributors/dcarpintero/deep-learning-notebooks.svg)](https://GitHub.com/dcarpintero/deep-learning-notebooks/graphs/contributors/)
[![GitHub issues](https://img.shields.io/github/issues/dcarpintero/deep-learning-notebooks.svg)](https://GitHub.com/dcarpintero/deep-learning-notebooks/issues/)
[![GitHub pull-requests](https://img.shields.io/github/issues-pr/dcarpintero/deep-learning-notebooks.svg)](https://GitHub.com/dcarpintero/deep-learning-notebooks/pulls/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)

[![GitHub watchers](https://img.shields.io/github/watchers/dcarpintero/deep-learning-notebooks.svg?style=social&label=Watch)](https://GitHub.com/dcarpintero/deep-learning-notebooks/watchers/)
[![GitHub forks](https://img.shields.io/github/forks/dcarpintero/deep-learning-notebooks.svg?style=social&label=Fork)](https://GitHub.com/dcarpintero/deep-learning-notebooks/network/)
[![GitHub stars](https://img.shields.io/github/stars/dcarpintero/deep-learning-notebooks.svg?style=social&label=Star)](https://GitHub.com/dcarpintero/deep-learning-notebooks/stargazers/)

## 01. Annotated Neural Network Classifier

This notebook implements in Python the essential modules required to build and train a multilayer perceptron that classifies garment images. In particular, it delves into the fundamentals of `approximation`, `non-linearity`, `regularization`, `gradients`, and `backpropagation`. Additionally, it explores the significance of `random parameter initialization` and the benefits of `training in mini-batches`.

By the end, you will be able to construct the building blocks of a neural network from scratch, understand how it learns, and deploy it to HuggingFace Spaces to classify real-world garment images.

[![Blog](https://img.shields.io/badge/Read-Blog-orange)](https://huggingface.co/blog/dcarpintero/building-a-neural-network-for-image-classification)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dcarpintero/nn-image-classifier/blob/main/nb.image.classifier.ipynb)
[![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-sm-dark.svg)](https://huggingface.co/spaces/dcarpintero/fashion-image-recognition)

<p align="center">
  <img src="https://cdn-uploads.huggingface.co/production/uploads/64a13b68b14ab77f9e3eb061/bvC2A3Cb2zn81h_neojtH.png">
</p>

<p align="center">Garment Classifier deployed to HuggingFace Spaces</p>

## 02. Quantization

Quantization is a method used to reduce the computational complexity  and memory footprint of a model by representing their weights and activations with low-precision data types like 8-bit integer, instead of the usual 32-bit floating point. This optimization results in less memory storage, and faster operations like matrix multiplication, which is a fundamental operation in the inference process.

The intuition behind quantization is that we can represent floating-point values in a tensor by mapping their range [max, min] into a smaller range [-128, 127], and then linearly distribute all values in between.


