from pulp import LpProblem, LpMaximize, LpVariable, lpSum
import pandas as pd

# Sample data: each row is a policy with predicted cancellation probability for various price changes
# and associated premium increase per price change level
# You'd typically generate this via your LSTM embeddings + pricing model

# Example input DataFrame
# Each row represents a policy with options: (price_change, premium_gain, cancellation_prob)
data = [
    {'policy_id': 'A', 'options': [(0.0, 0, 0.01), (0.05, 20, 0.05), (0.10, 40, 0.15)]},
    {'policy_id': 'B', 'options': [(0.0, 0, 0.01), (0.04, 15, 0.03), (0.08, 30, 0.10)]},
    {'policy_id': 'C', 'options': [(0.0, 0, 0.01), (0.03, 10, 0.05), (0.06, 20, 0.08)]},
]

# Global cancellation probability cap (based on risk tolerance)
cancellation_thresholds = {'A': 0.10, 'B': 0.08, 'C': 0.06}

# Create the optimization problem
prob = LpProblem("Personalized_Pricing_Optimization", LpMaximize)

# Variables: one binary variable per policy-price option
variables = {}
for policy in data:
    for i, (pct, premium, cancel_prob) in enumerate(policy['options']):
        var_name = f"x_{policy['policy_id']}_{i}"
        variables[var_name] = LpVariable(var_name, cat='Binary')

# Objective: maximize total premium gain
prob += lpSum(
    variables[f"x_{policy['policy_id']}_{i}"] * option[1]
    for policy in data
    for i, option in enumerate(policy['options'])
)

# Constraints: only one pricing option per policy
for policy in data:
    prob += lpSum(variables[f"x_{policy['policy_id']}_{i}"] for i in range(len(policy['options']))) == 1

# Constraints: stay below each policy's cancellation risk threshold
for policy in data:
    pid = policy['policy_id']
    prob += lpSum(
        variables[f"x_{pid}_{i}"] * option[2]
        for i, option in enumerate(policy['options'])
    ) <= cancellation_thresholds[pid]

# Solve the problem
prob.solve()

# Display selected pricing options
for var in prob.variables():
    if var.varValue == 1:
        print(f"Selected: {var.name}")

# Total optimized premium increase
print(f"Total Premium Gain: ${prob.objective.value():.2f}")
