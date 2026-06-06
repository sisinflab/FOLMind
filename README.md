# FOLMind: Logic-to-Narrative Profile Embeddings for Collaborative Filtering

**FOLMind** is a logic-to-narrative representation framework that transforms mined First-Order Logic (FOL) rules into narrative profiles, bridging symbolic relational knowledge and continuous semantic representations for collaborative recommendation. 

LLM-based recommender approaches often rely solely on local histories or metadata, underutilizing dataset-level regularities encoded in the interaction graph. FOLMind addresses this gap by creating a symbolic summary of the dataset's structure through FOL rules, verbalizing them into natural-language profiles, and infusing this knowledge into collaborative backbones.

This repository contains the full implementation of the FOLMind pipeline, from data processing and KG construction to rule mining, LLM-based narrative profile generation, semantic embedding, and hybrid recommendation experiments.

## 🔍 Overview
FOLMind performs **semantic smoothing**: it translates discrete logical implications into natural-language profile context before embedding them into a continuous semantic space.

The framework operates through a five-stage pipeline:
1.  **Knowledge Graph Construction**: Builds a unified, dataset-specific KG by mapping items to URIs in external knowledge graphs (e.g., DBpedia) and retrieving semantic neighborhood metadata via multi-hop SPARQL traversals.
2.  **Symbolic Rule Extraction**: Employs the AMIE rule miner to automatically extract Horn-clause rules from the KG (e.g., `watched(u, Interstellar) ∧ watched(u, Gravity) → prefers(u, sci-fi)`), providing symbolic evidence of recurring relational patterns.
3.  **Narrative Profile Generation**: Uses an LLM (e.g., **Gemini 2.5 Flash Lite**) with structured prompts to generate concise, narrative user and item profiles based on histories, metadata, and the applicable mined FOL rules.
4.  **Semantic Embedding Generation**: Encodes the generated narrative profiles into a dense semantic space using models like **Qwen3-Embedding-4B**.
5.  **Backbone Integration**: Integrates the resulting semantic representations into classical and neural collaborative recommendation models through backbone-specific strategies:
    *   **SpaceInit (for NeuMF)**: Initializes user and item representations by projecting the semantic embeddings into the neural model's latent space.
    *   **Hybrid-Blend (for MF)**: Computes semantic similarity from the embeddings, propagates it through the semantic neighborhood, and linearly combines it with the collaborative MF score at inference time.

### ✨ Key Features
*   **End-to-End Logic-to-Narrative Pipeline**: A complete workflow from raw data to hybrid recommendation, including crawling, KG construction, rule mining, and LLM narrative generation.
*   **Symbolic Profile Enrichment**: Uses mined FOL rules as structured evidence to make LLM-generated profiles aware of global dataset semantics, improving upon narrative-only profiles.
*   **Flexible Collaborative Integration**: Implements model-specific integration strategies (`SpaceInit` and `Hybrid-Blend`) to enhance existing collaborative recommendation paradigms.
*   **Empirical Validation**: Codebase to reproduce the experiments on **MovieLens-1M** and **Book Genome** datasets, demonstrating nDCG gains up to +30.51%.

---

## Table of Contents

- [Reproducing Experimental Results](#reproducing-experimental-results)
- [Installation](#installation)
- [Usage](#usage)
- [Framework Architecture](#framework-architecture)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)

---
## 📊 Reproducing Experimental Results

To replicate the experimental results from the FOLMind paper (evaluating NeuMF and MF on MovieLens 1M and Book Genome), follow these steps:

1.  **Install the Environment**
    Follow all the steps in the [Installation](#installation) section below to set up the `folm_env` conda environment, install all dependencies, and configure your API key.

2.  **Run the Full Pipeline and Experiments**
    The following commands will execute the entire workflow: data processing, rule mining, embedding generation, and finally, the recommendation experiments.

    *   **Step 1: Run the Data Processing and Embedding Generation Pipeline**
        This command crawls the data, builds the KG, mines rules with AMIE, generates Gemini LLM-based narrative profiles, and encodes them using DEFAULT MODELS. To use `Qwen3-4B` please refer to `variantHandler.py`.
        ```bash
        python run.py --config ml1m_base_settings_baseline.yml --run-all
        ```

    *   **Step 2: Run the MF Experiments (Hybrid-Blend)**
        This evaluates Matrix Factorization (MF) enhanced with the generated embeddings using the Hybrid-Blend score-fusion strategy.
        ```bash
        python run_experiments.py -c ml1m_base_settings_qwen4BFOL.yml
        ```

    *   **Step 3: Run the NeuMF Experiments (SpaceInit)**
        This evaluates the Neural Matrix Factorization (NeuMF) models using the `SpaceInit` strategy. *(Note: Ensure your neural config file points to the desired Qwen3 embeddings generated in Step 1)*.
        ```bash
        python run_neural_experiments.py --config ml1m_optuna_neural_baseline.yml
        python run_neural_experiments.py --config ml1m_optunaNewNMF_LLMEmb_QWEN4.yml
        ```

---

## 📥 Installation

### Prerequisites
*   [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
*   Git

### 1. Clone the Repository
```bash
git clone https://github.com/.../FOLMind.git
cd FOLMind
```

### 2. Create and Activate the Conda Environment
This command creates a new Conda environment named `folm_env` using the provided `environment.yml` file.
```bash
conda env create -f environment.yml
conda activate folm_env
```

### 3. Install PyTorch and Dependencies
Choose the appropriate command based on your hardware (NVIDIA GPU or CPU-only).

#### Option A: With NVIDIA GPU (CUDA 12.1 Support)
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.5.0+cu121.html
```

#### Option B: CPU-Only Installation
```bash
pip3 install torch torchvision torchaudio
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.6.0+cpu.html
```
*Note regarding Model Compatibility: Selecting Option B may result in compatibility issues with certain text embedding models due to limitations within the "transformers" library.*

### 4. Verify the Installation
Run the following command to ensure all dependencies are correctly installed.
```bash
pip check
```
The expected output is: `No broken requirements found.` *(If packages were skipped during environment creation, install them manually via pip).*

### 5. Set Up Google AI API Key
FOLMind uses the Google Gemini API (specifically Gemini 2.5 Flash Lite) for narrative profile generation. You must configure your API key within the relevant YAML configuration file in the `configs/` directory.

Find the `LLMEmb` section in your chosen config file (e.g., `configs/ml1m_base_settings.yml`) and add your key(s):
```yaml
LLMEmb:
  api_keys:
    - "YOUR_GOOGLE_AI_API_KEY_HERE"
```
You can obtain a key from your [Google AI Studio account](https://aistudio.google.com/app/apikey).

### Updating the Environment
If the `environment.yml` file is updated, you can update your existing environment:
```bash
conda activate folm_env
conda env update --file environment.yml --prune
```

---

## 🚀 Usage

The framework is executed via three main scripts: `run.py` (for data/KG/LLM pipeline), `run_experiments.py` (for MFevaluation), and `run_neural_experiments.py` (for NeuMF evaluation).

### 1. Configure Your Run
Modify a YAML configuration file in the `configs/` directory. This file controls the dataset paths (e.g., MovieLens 1M, Book Genome), 10-core filtering, AMIE extraction parameters (atom depth, support), and LLM profile generation templates.

### 2. Run the Data & Embedding Pipeline
The `run.py` script executes the semantic smoothing process. You can run the entire pipeline or specific stages:

*   **Run all stages (recommended for the first run):**
    ```bash
    python run.py --config your_config.yml --run-all
    ```

*   **Run specific stages individually:**
    ```bash
    # Stage 1: Build the unified knowledge graph (local data + SPARQL DBpedia traversal)
    python run.py --config your_config.yml --run-crawler

    # Stage 2: Mine First-Order Logic (FOL) rules with AMIE
    python run.py --config your_config.yml --run-amie

    # Stage 3: Generate logic-to-narrative profiles via Gemini LLM
    python run.py --config your_config.yml --run_llm_calls

    # Stage 4: Encode text profiles into dense semantic embeddings (e.g., Qwen3)
    python run.py --config your_config.yml --run-llm-pipeline
    ```

### 3. Run Recommendation Experiments (Stage 5)
Once embeddings are generated, integrate them into the backbones.

*   **Hybrid-Blend for Matrix Factorization (MF):**
    ```bash
    python run_experiments.py -c your_config.yml
    ```
    Trains the MF backbone, computes top-k semantic neighborhood similarity from normalized profile embeddings, and blends the collaborative scores with semantic predictions at inference time.

*   **SpaceInit for Neural Matrix Factorization (NeuMF):**
    ```bash
    python run_neural_experiments.py --config your_neural_config.yml
    ```
    Uses semantic profile embeddings to project and warm-start NeuMF's latent representation space (`p_u` and `q_i`) before joint optimization.

---

## 🏛️ Framework Architecture

The codebase is organized into modular components reflecting the paper's methodology:

*   **`crawler/`**: Constructs the KB-agnostic Knowledge Graph via URI mapping and multi-hop SPARQL traversal to retrieve semantic metadata (e.g., genres, directors, topics).
*   **`amie/`**: Interfaces with the AMIE system to discover dataset-level Horn-clause rules (`R`) over the graph `G`.
*   **`Emb/`**: The core logic-to-narrative module. Handles rule filtering, prompt generation, Gemini API interactions (`LLMEmbAPI`), parsing reasoning traces, and dense embedding encoding using models like Qwen3-Embedding-4B.
*   **`recommender/`**: Implements collaborative scorers (e.g., `MF_Elliot`) and the `Folminder` handler for applying `Hybrid-Blend` (score-level fusion) operations.
*   **`neural_recommender/`**: Contains PyTorch implementations of `NeuMF` architectures, the trainer, and the projection mechanisms required for the `SpaceInit` strategy.
*   **`configs/` & `neural_configs/`**: YAML directories managing hyperparameters optimized via Optuna (e.g., semantic neighbor counts, alpha blending weights).

---

## ⚙️ Configuration

All pipeline settings are exposed via YAML:

*   **Main Config (`configs/`)**: Defines the dataset, sparsity filters, AMIE rule thresholding (head coverage, support), LLM context integration templates, and MF evaluation parameters.
*   **Crawler Config**: Defines semantic enrichment rules, SPARQL endpoints (e.g., DBpedia), and graph relation mappings (e.g., `has_genre`, `directed_by`).
*   **Neural Config (`neural_configs/`)**: Controls NeuMF training dynamics (BPR sampling, AdamW, early stopping) and specific `SpaceInit` projection dimensions.

---

## 🤝 Contributing

Contributions to expand FOLMind to new datasets, integration strategies, or rule miners are welcome! Please fork the repository, create a new branch for your feature, and submit a pull request.

---

## 📄 License
This project is licensed under the Apache 2.0 License. See the [LICENSE](LICENSE) file for details.

---