import pandas as pd

# Load your dataset
df = pd.read_csv('/mnt/data/project_benchmark_data_ce.csv')

# Display the first few rows of the dataset
st.write("Dataset Overview:")
st.write(df.head())  # Show first 5 rows

# Display the columns of the dataset
st.write("Columns in Dataset:")
st.write(df.columns)  # Show column names
