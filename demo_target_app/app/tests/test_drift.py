import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from demo_target_app.app.services.drift import calculate_embedding_drift
from demo_target_app.app.services.drift import calculate_query_distribution_drift


# 1. Simulate "Baseline" data (e.g., last week's document embeddings)
# We generate 5 random 768-dimensional vectors (simulating nomic-embed-text outputs)
np.random.seed(42)
baseline_data = np.random.normal(loc=0.0, scale=0.1, size=(15, 768)).tolist()

# 2. Simulate "Recent" data that is highly similar (Should NOT trigger drift)
recent_stable = np.random.normal(loc=0.02, scale=0.1, size=(15, 768)).tolist()

# 3. Simulate "Recent" data that shifted wildly (SHOULD trigger drift)
recent_drifted = np.random.normal(loc=0.8, scale=0.1, size=(15, 768)).tolist()

print("TEST 1: Stable Knowledge Base")
result_1 = calculate_embedding_drift(baseline_data, recent_stable)
for key, value in result_1.items():
    print(f"  {key}: {value}")

print("\nTEST 2: Drifted Knowledge Base")
result_2 = calculate_embedding_drift(baseline_data, recent_drifted)
for key, value in result_2.items():
    print(f"  {key}: {value}")


print("\nTESTING SIGNAL 3: Query Distribution Drift (KS-Test)")
esult_stable = calculate_query_distribution_drift(baseline_data, recent_stable)
print(f"  Stable User Behavior -> is_drifted: {recent_stable['is_drifted']} (Score: {recent_stable['drift_score']})")

result_shifted = calculate_query_distribution_drift(baseline_data, recent_drifted)
print(f" Shifted User Behavior -> is_drifted: {result_shifted['is_drifted']} (Score: {result_shifted['drift_score']})")