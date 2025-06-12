import pandas as pd

# Define the path to your CSV file
csv_file_path = r'C:\Users\lndnc\OneDrive\Desktop\AI\DRWkaggle\test.csv' # Make sure this file is in the same directory or provide the full path

try:
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_file_path)

    # Print the column names
    print("Columns in the DataFrame:")
    first_column_name = df.columns[0]
    print(f"The first column in the DataFrame is: {first_column_name}")

    for col in df.columns:
        print(col)

except FileNotFoundError:
    print(f"Error: The file '{csv_file_path}' was not found.")
except Exception as e:
    print(f"An error occurred: {e}")