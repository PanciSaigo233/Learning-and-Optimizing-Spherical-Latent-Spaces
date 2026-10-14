# Learning and Optimizing Spherical Latent Spaces

**Master's Thesis — Computer Science and Engineering, Politecnico di Milano**  
**Author:** Jiaxiang Yi  
**Research areas:** Deep Learning · Generative Models · Representation Learning · Hyperspherical Geometry · Semantic Embeddings

## Overview

This repository collects Jupyter notebooks developed during my master's thesis research on **learning, evaluating, and optimizing latent representations**, with a particular focus on **hyperspherical latent spaces**.

The work investigates how the geometry of a latent space affects image reconstruction, generative sampling, representation quality, and the preservation of semantic relationships. In addition to experiments with variational autoencoders (VAEs), the repository explores spherical word representations and geometry-aware transformations of semantic relations.

The notebooks are research experiments rather than a single end-to-end software package. They cover different model variants, datasets, metrics, and exploratory studies.

## Research Questions

- How does constraining latent representations to a unit hypersphere affect generative modeling compared with conventional Gaussian latent spaces?
- Can geometry-aware sampling and regularization improve the organization and usability of latent representations?
- How do different latent-space designs compare across visual datasets and quantitative generative metrics?
- Can word representations learned from context be embedded on a hypersphere and used for semantic retrieval and relation transport?

## Research Tracks

### 1. Variational Autoencoders and Hyperspherical Representations

Experiments investigate multiple latent-space formulations, including:

- **Standard Gaussian VAE:** a baseline using a Euclidean latent representation and Gaussian sampling.
- **Spherical VAE:** representations constrained to the unit hypersphere, with geometry-aware sampling and latent-space regularization.
- **von Mises–Fisher (vMF) VAE:** a directional-distribution baseline for comparison.

The research includes exploration of geodesic interpolation, latent-space organization, reconstruction quality, sampling behavior, and regularization strategies.

**Related notebooks (examples):**

| Notebook / filename pattern | Experiment |
| --- | --- |
| `autoencoder-spherical-annealing-sum.ipynb` | Spherical autoencoder and annealing/regularization exploration |
| `VAE-CelebA_64_Color_00.ipynb` | VAE experiments with CelebA face images |
| `VAE_FashionMNIST_BN_00.ipynb` | VAE experiments on Fashion-MNIST |
| `VAE_BloodMNIST_BN_00.ipynb` | VAE experiments on BloodMNIST |
| `VAE_BloodMNIST_Color_00.ipynb` | Color-image experiments on BloodMNIST |
| `VAE_retinamnist_64_Color_00.ipynb` | VAE experiments on RetinaMNIST |
| `VAE-pkg-tp53-precursor-lesions.ipynb` | Exploratory VAE notebook concerning TP53/precursor-lesion data |

> The listed notebooks correspond to filenames visible in the repository screenshot. Their individual configurations and results may differ; refer to the notebook code and outputs for the exact experimental settings.

### 2. Quantitative Generative Evaluation

The repository includes notebooks dedicated to comparing generative models using **Fréchet Inception Distance (FID)** across different datasets.

| Notebook | Dataset / focus |
| --- | --- |
| `fid-threemodels-Flower102.ipynb` | Flowers102 |
| `fid-threemodels-bloodmnist.ipynb` | BloodMNIST |
| `fid-threemodels-cifar-100.ipynb` | CIFAR-100 |
| `fid-threemodels-cifar10.ipynb` | CIFAR-10 |
| `fid-threemodels-svhn.ipynb` | SVHN |

These experiments support comparisons of generated-image quality under different modeling assumptions. FID should be interpreted alongside reconstruction metrics and qualitative inspection, since it does not by itself measure semantic structure or representation usefulness.

### 3. Spherical Word Embeddings and Semantic Relations

A second research direction investigates **spherical representations for language**. Here, contextual information is mapped to latent vectors on the unit hypersphere, and word-level representations can be formed by aggregating representations obtained from different contexts.

Key concepts explored include:

- **Context-based representation learning** using the Text8 corpus.
- **Word-level spherical prototypes** obtained from context-dependent latent embeddings.
- **Nearest-neighbor semantic retrieval** in a hyperspherical embedding space.
- **Relation vectors in tangent spaces**, calculated using the spherical logarithmic map.
- **Parallel transport** of semantic relations between different points on the sphere.
- **Exponential mapping** to return transported tangent vectors to the sphere.
- **BATS benchmark evaluation** of semantic and morphological relations.

**Related notebooks:**

| Notebook / filename pattern | Experiment |
| --- | --- |
| `Demo 1 -direct-spherical-context-encoder-dec...` | Context encoder–decoder for spherical word representations |
| `Demo 2 - Character-level Spherical VAE for wor...` | Character-level spherical VAE exploration |
| `Demo 3 — Ontology Concept Vector.ipynb` | Ontology/concept-vector exploration |
| `direct-spherical-context-text8-relation-transpor...` | Text8 embeddings and spherical relation transport |
| `bats-country-test.ipynb` | Country/capital semantic-relation evaluation |

> Some filenames are abbreviated here because the screenshot truncates their full names. Locate the corresponding notebooks by their visible prefixes.

#### Geometry-aware relation transport

For unit vectors \(x,y\in\mathbb{S}^{d-1}\), the spherical logarithmic map represents a relationship as a tangent vector at \(x\):

\[
\operatorname{Log}_x(y) = \frac{\theta}{\sin\theta}(y-\cos\theta\,x),
\qquad \theta=\arccos(x^\top y).
\]

The vector can be **parallel-transported** to the tangent space at another point \(z\), then mapped back to the sphere using the **exponential map**:

\[
\hat{y} = \operatorname{Exp}_z\!\left(\operatorname{PT}_{x\rightarrow z}(\operatorname{Log}_x(y))\right).
\]

This provides a geometric way to test whether a learned semantic relationship can be transferred to a new context. Numerical implementations require special handling near identical or antipodal points.

## Repository Organization

The repository currently stores most experiments as individual Jupyter notebooks in its root directory:

```text
.
├── README.md
├── VAE-*.ipynb                        # VAE experiments on different datasets
├── autoencoder-spherical-*.ipynb     # Spherical latent-space explorations
├── fid-threemodels-*.ipynb           # Generative evaluation notebooks
├── direct-spherical-context-*.ipynb  # Context-based spherical word embeddings
├── bats-country-test.ipynb           # Semantic relation benchmark
└── Demo *.ipynb                      # Additional representation-learning demos
```

The structure above is a **logical overview**, not an exact inventory of every file or directory.

## Methodology

Across the notebooks, the general experimental workflow is:

1. **Prepare data:** load and preprocess images or text, and construct training/evaluation splits.
2. **Build representations:** train a baseline or geometry-aware model and inspect its latent vectors.
3. **Optimize:** evaluate different objectives, regularization settings, sampling approaches, or network designs.
4. **Evaluate:** compare quantitative metrics and qualitative behavior across experiments.
5. **Analyze latent geometry:** examine interpolation, neighborhoods, and—in the language experiments—semantic relation transport.

The image-generation and semantic-representation tracks use distinct datasets, architectures, and evaluation protocols. Results should therefore be compared **within the corresponding experiment**, not as interchangeable benchmarks.

## Getting Started

### Prerequisites

- Python 3
- JupyterLab or Jupyter Notebook
- A compatible machine-learning environment (GPU recommended for training-intensive experiments)

Depending on the notebook, you may need packages such as **PyTorch**, **NumPy**, **Matplotlib**, **scikit-learn**, **torchvision**, and dataset- or metric-specific libraries. Check each notebook's imports and setup cells for its actual dependencies.

### Run a notebook

```bash
git clone <YOUR_REPOSITORY_URL>
cd <YOUR_REPOSITORY_DIRECTORY>
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate  # Windows PowerShell
pip install jupyterlab
jupyter lab
```

Open the relevant notebook and install its remaining requirements as specified by its imports. Update dataset paths, device settings, and output directories before running the cells.

**Reproducibility note:** Some notebooks may assume locally stored datasets, trained checkpoints, or notebook-specific package versions. This repository should not be interpreted as providing a fully automated reproduction pipeline unless those assets and configurations are also available.

## Evaluation and Interpretation

The experiments include several complementary evaluation perspectives:

| Area | Examples of evaluation |
| --- | --- |
| Image generation | FID, visual quality of generated samples |
| Reconstruction | Reconstruction error and reconstructed-image inspection |
| Latent geometry | Vector norms, interpolation, neighborhood structure |
| Semantic embeddings | Nearest-neighbor retrieval, coverage, target-word ranking |
| Semantic relations | BATS and country–capital relation transport |

For reproducible numerical comparisons, consult the specific notebook outputs, model checkpoints, data splits, and metric implementations. This README does not present a single aggregate score because the experiments span different datasets and tasks.

## Motivation and Main Contribution

The central theme of this work is that **latent-space geometry is a design choice**, rather than merely an implementation detail. By investigating Euclidean and spherical representations across generative and semantic tasks, the research explores how geometry influences sampling, interpolation, neighborhood relationships, and downstream use of learned representations.

The repository documents both successful experiments and exploratory alternatives, reflecting the iterative nature of thesis research.

## Academic Context

This repository contains experiments associated with the master's thesis **“Learning and Optimizing Spherical Latent Spaces”** at **Politecnico di Milano**.

**Author:** Jiaxiang Yi  
**Supervisors:** Prof. Pietro Pinoli and Dr. Stefano Perna

## Notes

- The notebooks are intended for academic research and experimentation.
- Dataset licenses and usage conditions remain those of the original dataset providers.
- Different notebooks may use different preprocessing procedures, hyperparameters, and dependency versions.
- Filenames and file layouts may change as the repository is reorganized.

---

*Research focus: improving the way complex information is represented, sampled, and transformed through the geometry of learned latent spaces.*
