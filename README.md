# TRTL

**Temporal Link Prediction on Interval-based Knowledge Graphs via Explainable Temporal Relation Tree-Graphs**

## Environment Setup

Install the required dependencies:

```sh
pip install -r requirements.txt
```

This project has been tested with:

* Python ≥ 3.8
* CUDA-enabled GPU

---

##  Quick Start

Run TRTL on **YAGO11k** or **WIKIDATA12k**:

```sh
cd src
python main.py --dataname yago --maxL 5 --TopK 300
python main.py --dataname wiki --maxL 3 --TopK 500
```

### **Dataset Options**

* `--dataname yago` → **YAGO11k** (interval-based temporal KG)
* `--dataname wiki` → **WIKIDATA12k** (interval-based temporal KG)

### **Key Arguments**

* `--maxL`
  Maximum expansion depth of the Temporal Relation Tree-Graph (TRTG).
  Higher values enable deeper multi-hop reasoning.

* `--TopK`
  Number of branches retained per layer during pruning.
  Controls the reasoning width and ensures linear-time graph expansion.

---

## Viewing Results

Model outputs and evaluation results are saved in:

```
results.log
```
