# PhosphoNetworks Manuscript Code
[![Package Install](https://github.com/saezlab/phosphonetworks/actions/workflows/package-install.yml/badge.svg)](https://github.com/saezlab/phosphonetworks/actions/workflows/package-install.yml)
[![Paper Pipeline](https://github.com/saezlab/phosphonetworks/actions/workflows/paper-pipeline.yml/badge.svg)](https://github.com/saezlab/phosphonetworks/actions/workflows/paper-pipeline.yml)

This repository contains the high-level code and notebooks that accompany the manuscript:  

*Garrido-Rodriguez et al., “Unexplored Signaling Pathways: Escaping the Streetlight Effect with Phosphoproteomics and Kinase–Substrate Interactions” (2025)*

The `phosphonetworks` Python module provides all functions required to reproduce the analyses and figures in the paper.  
It also includes convenient wrappers for **kinase activity estimation** and **kinase–kinase network inference**.

---

## Installation

Clone the repository and install the package with pip:

```bash
pip install .
```

Additional dependencies are required for specific analyses and visualizations:

### 1. Gurobi
Gurobi is used to solve the Rooted Prize-Collecting Steiner Tree problem described in the manuscript.  
Academic licenses are available free of charge.  
Request a license at: [https://www.gurobi.com/](https://www.gurobi.com/)

### 2. Graphviz
Graphviz is needed to generate network plots.  
Download and install it from: [https://graphviz.org/download/](https://graphviz.org/download/)

### 3. MMSeqs 
MMSeqs is used to compute the kinase-kinase sequence similarity scores.
Download and install it from: [https://mmseqs.com/](https://mmseqs.com/)

---

## Notebooks

Two Jupyter notebooks are provided:

1. **Paper pipeline** – Downloads the manuscript’s data from Zenodo and reproduces all analyses and figures.  
   [Open the notebook](./paper_pipeline.ipynb)

2. **Analysis tutorial** – Demonstrates how to estimate kinase activity using different kinase–substrate networks from the paper and how to infer a kinase–kinase signaling network.  
   [Open the notebook](./analysis_tutorial.ipynb)
