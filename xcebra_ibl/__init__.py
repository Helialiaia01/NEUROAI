"""
xCEBRA-IBL: Explainable Contrastive Learning for IBL Neural Responses
=====================================================================

Replaces the linear Reduced-Rank Regression (RRR) encoding model from
"Rarely Categorical" (Wang et al.) with xCEBRA's regularized contrastive
learning to re-test clustering of brain-region response profiles.

Pipeline:
    1. data/   – Download & preprocess IBL data (same as brainwide-RRR)
    2. models/ – xCEBRA training with per-variable attribution maps
    3. analysis/ – Selectivity clustering & comparison with RRR results
"""
