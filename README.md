# Zomathon Problem Statement 2: Cart Super Add-On (CSAO) Recommendation System

## 1. Project Overview
This repository contains our end-to-end solution for **Problem Statement 2**. We have built a highly scalable, real-time recommendation system to improve Cart-to-Order (C2O) ratios and Average Order Value (AOV) by intelligently suggesting complementary cart add-ons. 

Our solution leverages a fast **LightGBM LambdaMART** learning-to-rank model for inference, enriched by advanced **Large Language Model (LLM)** semantic embeddings to solve the "cold-start" problem for new food items.

## 2. Directory Structure & Code Modules

The codebase is organized into a modular, production-ready MLOps structure:

```text
/zomathon
│
├── data/                   # Generated synthetic CSVs and Numpy Embeddings
├── models/                 # The compiled LightGBM ranker (.pkl)
├── Zomaton_PS2_Submission.pdf   # Our official Executive Summary Report
│
└── src/                    # Source Code Directory
    ├── data/               # Scripts 1-4: Raw data generation (Entities, Items, Sessions)
    ├── features/           # Script 5: Feature Engineering pipeline
    ├── models/             # Script 6: Model Training (LightGBM Ranker)
    ├── api/                # Script 7-8: FastAPI Server and Latency Testing
    └── reports/            # Script 9: PDF/HTML generation scripts
```

## 3. How to Run the Project

### Prerequisites
- Python 3.9+
- A virtual environment is highly recommended.
- Install dependencies: `pip install pandas numpy scikit-learn lightgbm fastapi uvicorn requests sentence-transformers`

### Phase 1: Data Generation & Preprocessing
If you wish to re-generate the synthetic dataset from scratch, run the scripts in order:
1. `python src/data/1_generate_entities.py` (Generates Users & Restaurants)
2. `python src/data/2_generate_items.py` (Generates 20,000 Menu Items)
3. `python src/data/3_generate_embeddings.py` (Creates LLM Semantic Embeddings)
4. `python src/data/4_generate_sessions.py` (Simulates cart events and biological habits)
5. `python src/features/5_feature_engineering.py` (Creates the ML training matrix)

### Phase 2: Model Training
To retrain the LightGBM Ranker on the engineered data:
* `python src/models/6_train_model.py` (Outputs `lgbm_ranker.pkl` to `/models`)

### Phase 3: Serving & Latency Testing (< 300ms SLA)
To boot the recommendation engine server and verify latency:
1. Start the API: `python src/api/7_api_server.py`
2. In a new terminal, run the latency benchmark: `python src/api/8_test_latency.py`
* Note: Our benchmarks consistently demonstrate an inference time of **~35ms**, well within the 300ms constraint.

## 4. Evaluation & Business Impact
Our rigorous offline evaluation (Temporal Train/Test split) produced the following metrics:
- **Test AUC:** `0.841`
- **NDCG@1 (Top-1 Acceptance Rate):** `0.78`
- **Simulated Baseline AOV:** `~₹340.00`
- **Projected AOV Lift:** By surfacing highly-contextual Add-ons driven by exact current-cart context (Incomplete Meals), Geographic Zones, and Hyper-Personalized Temporal states (Cyclical Time bounds), we project a **~4-7% organic increase** in Gross Order Value purely through mathematical relevance mapping.

Detailed methodology, design trade-offs, and A/B testing frameworks are strictly documented in the accompanying **Zomaton_PS2_Submission.pdf**.
