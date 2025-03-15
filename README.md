# Dynamic Modeling of Patients, Modalities, and Tasks via Multi-modal Multi-task Mixture of Experts

This repository contains the code implementation for the ICLR 2025 poster titled **"Dynamic Modeling of Patients, Modalities and Tasks via Multi-modal Multi-task Mixture of Experts."** Our work sits at the intersection of artificial intelligence and medicine, aiming to advance the state-of-the-art in medical imaging and patient-specific modeling.

## Overview

In this project, we introduce a multi-modal, multi-task mixture of experts framework designed for dynamic modeling across various medical domains. The current release focuses on the Mammography domain and includes:

- **Model Implementation:** A custom Vision Transformer-based architecture with a Mixture-of-Experts design.
- **Datasets:** Code for dataset processing, including preprocessing and train-test split CSV generation (hosted on Hugging Face).
- **Training and Testing Pipelines:** End-to-end training, evaluation, and checkpointing scripts.
- **Analysis Pipelines:** Preliminary analysis code for exploring model performance and interpretability.

## Citation

If you find our work or code useful for your research, please consider citing our paper in APA format:

> Wu, C., Shuai, Z., Tang, Z., Wang, L., & Shen, L. (2025). Dynamic Modeling of Patients, Modalities and Tasks via Multi-modal Multi-task Mixture of Experts. In *The Thirteenth International Conference on Learning Representations*.

## Release Timeline

### March 13, 2025
- **Released:** Model, dataset, training and testing code for the Mammography domain.
- **Additional Resources:** 
  - Dataset preprocessing scripts and train-test split CSVs available on [Hugging Face](https://huggingface.co).
  - Partial release of analysis pipelines.

## Planned Future Releases

### By May 2025
- **Upcoming:**
  - Full release of the remaining analysis code (e.g., Partial Information Decomposition). For now, please refer to the [PID repository](https://github.com/pliang279/PID) for reference.
  - Release of model weights, a demo, and a YouTube tutorial video.

## Getting Started

To get started with our code:
**Clone the Repository:**

   git clone https://github.com/tangzhengxu2001/m4oe.git
   pip install -r dependency.txt
