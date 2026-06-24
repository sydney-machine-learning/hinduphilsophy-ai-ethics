import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)

# ------------------------------------------------------------
# Synthetic simulation of AI decision options
# ------------------------------------------------------------

n_cases = 500
n_actions_per_case = 5

rows = []

for case_id in range(n_cases):
    for action_id in range(n_actions_per_case):
        utility = np.random.uniform(0.3, 1.0)
        harm = np.random.beta(2, 5)              # mostly low, sometimes high
        bias = np.random.beta(2, 4)              # unfairness risk
        justification = np.random.uniform(0.2, 1.0)
        transparency = np.random.uniform(0.2, 1.0)
        accountability = np.random.uniform(0.2, 1.0)
        reasoning_error = np.random.beta(2, 5)

        rows.append({
            "case_id": case_id,
            "action_id": action_id,
            "utility": utility,
            "harm": harm,
            "bias": bias,
            "justification": justification,
            "transparency": transparency,
            "accountability": accountability,
            "reasoning_error": reasoning_error
        })

df = pd.DataFrame(rows)

# ------------------------------------------------------------
# Objective functions based on the paper's formulation
# ------------------------------------------------------------

def utility_only_score(row):
    return row["utility"]

def ahimsa_score(row, lambda_h=1.2):
    # max E[R_task - lambda * H]
    return row["utility"] - lambda_h * row["harm"]

def nyaya_ahimsa_score(row, lambda_h=1.2, lambda_b=0.8, gamma_j=0.7, lambda_e=0.8):
    # Combines:
    # U - lambda*Bias + gamma*Justification
    # J + T + A - lambda*Error
    # R_task - lambda*Harm
    nyaya_component = (
        gamma_j * row["justification"]
        + 0.4 * row["transparency"]
        + 0.4 * row["accountability"]
        - lambda_e * row["reasoning_error"]
    )

    ahimsa_component = row["utility"] - lambda_h * row["harm"]

    fairness_penalty = lambda_b * row["bias"]

    return ahimsa_component + nyaya_component - fairness_penalty

df["score_utility_only"] = df.apply(utility_only_score, axis=1)
df["score_ahimsa"] = df.apply(ahimsa_score, axis=1)
df["score_nyaya_ahimsa"] = df.apply(nyaya_ahimsa_score, axis=1)

# ------------------------------------------------------------
# Select best action per case under each system
# ------------------------------------------------------------

def select_best_actions(score_column):
    idx = df.groupby("case_id")[score_column].idxmax()
    selected = df.loc[idx].copy()
    selected["system"] = score_column
    return selected

selected_utility = select_best_actions("score_utility_only")
selected_ahimsa = select_best_actions("score_ahimsa")
selected_nyaya_ahimsa = select_best_actions("score_nyaya_ahimsa")

results = pd.concat([
    selected_utility,
    selected_ahimsa,
    selected_nyaya_ahimsa
], ignore_index=True)

# ------------------------------------------------------------
# Summary metrics
# ------------------------------------------------------------

summary = results.groupby("system")[[
    "utility",
    "harm",
    "bias",
    "justification",
    "transparency",
    "accountability",
    "reasoning_error"
]].mean().round(3)

print("\nAverage selected-action performance by system:\n")
print(summary)

# ------------------------------------------------------------
# Plot 1: Utility vs Harm
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))
for system_name, group in results.groupby("system"):
    plt.scatter(group["utility"], group["harm"], alpha=0.5, label=system_name)

plt.xlabel("Utility / Task Performance")
plt.ylabel("Expected Harm")
plt.title("Utility-Harm Trade-off Across AI Decision Systems")
plt.legend()
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# Plot 2: Average harm by system
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))
summary["harm"].plot(kind="bar")
plt.ylabel("Average Expected Harm")
plt.title("Ahimsa Effect: Harm Reduction Across Systems")
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# Plot 3: Average justification by system
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))
summary["justification"].plot(kind="bar")
plt.ylabel("Average Justification Score")
plt.title("Nyaya Effect: Evidence-Based Justification Across Systems")
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# Plot 4: Average bias by system
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))
summary["bias"].plot(kind="bar")
plt.ylabel("Average Bias Risk")
plt.title("Fairness Effect: Bias Risk Across Systems")
plt.tight_layout()
plt.show()
