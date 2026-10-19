
import pandas as pd
import numpy as np

# Load the original dataset
df = pd.read_csv("data/customer_churn.csv")

# Create reproducible simulated dates
np.random.seed(42)

start_date = pd.Timestamp("2023-01-01")
end_date = pd.Timestamp("2025-12-31")

days = (end_date - start_date).days

random_days = np.random.randint(0, days + 1, size=len(df))

df["TransactionDate"] = (
    start_date + pd.to_timedelta(random_days, unit="D")
)

# Save as a new file
df.to_csv("data/customer_churn_with_dates.csv", index=False)

print("New dataset created successfully!")
print(df[["customerID", "TransactionDate", "MonthlyCharges"]].head())

