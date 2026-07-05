import numpy as np
from scipy.spatial.distance import euclidean
from scipy.stats import ks_2samp

def calculate_embedding_drift(baseline_embeddings: list[list[float]], recent_embeddings: list[list[float]], threshold_std: float = 2.0) -> dict:
    
    if len(baseline_embeddings)<5 or len(recent_embeddings) < 5:
        return {"is_drifted": False, "drift_score": 0, "reason":"Insufficient data for drift detection"}
    
    base_matirx = np.array(baseline_embeddings)
    recent_matirx = np.array(recent_embeddings)

    base_centroid = np.mean(base_matirx, axis=0)
    recent_centroid = np.mean(recent_matirx, axis=0)

    distance = euclidean(base_centroid, recent_centroid)

    base_distances = [euclidean(vec,base_centroid)  for vec in base_matirx]
    baseline_std = np.std(base_distances)

    dynamic_threshold = baseline_std + threshold_std
    is_drifted = bool(distance > dynamic_threshold)

    return{
        "is_drifted": is_drifted,
        "drift_score": round(float(distance),4),
        "threshold_limit": round(float(dynamic_threshold), 4),
        "reason": f"Centroid shifted ny {distance:.4f} (Allowed Variance: {dynamic_threshold:.4f})"
    }

# Kolmogorov-Smirnov test to detect statistical shifts in user query behavior
def calculate_query_distribution_drift(baseline_embeddings: list[list[float]], recent_embeddings: list[list[float]], alpha: float = 0.05) ->dict:
    if len(baseline_embeddings) < 10 or len(recent_embeddings)<10:
        return{"is_drifted": False, "drift_score":1.0, "reason": "Insufficient query volume for KS-Test"}
    
    base_matirx = np.array(baseline_embeddings)
    recent_matirx = np.array(recent_embeddings)

    num_dimensions = base_matirx.shape[1]
    significant_shifts = 0

    for d in range(num_dimensions):
        base_dim = base_matirx[:,d]
        recent_dim = recent_matirx[:,d]

        _, p_value = ks_2samp(base_dim, recent_dim) # ks_2samp calculates the p-value for this specific dimension

        if p_value <alpha:
            significant_shifts+=1
        
        drift_ratio = significant_shifts / num_dimensions

        is_drifted = drift_ratio > 0.10

        return{
            "is_drifted" : is_drifted,
            "drift_score": round(drift_ratio,4),
            "significant_dimensions_count": significant_shifts,
            "reason": f"Query distributions drifted in {drift_ratio * 100:.1f}% of semantic dimensions."
        }
    
