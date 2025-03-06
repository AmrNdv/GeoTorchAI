import pandas as pd

x_df = pd.read_csv("data/sat4_construction/x_test_sat4.csv", header=None)
print('loaded x')

# Select every fourth column
y_df = x_df.iloc[:, 3::4]

# Save the new DataFrame to a csv
y_df.to_csv("data/sat4_construction/y_test_sat4.csv", index=False, header=None)


