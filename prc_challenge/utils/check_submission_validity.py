import pandas as pd
from pandas.testing import assert_frame_equal

# GEMINI 2.5pro generated
# Used to check if the submission files have the right content


def compare_parquet_files(path1, path2, exclude_col="fuel_kg"):
    """
    Loads two parquet files and checks if they are identical,
    ignoring the specified column.
    """
    print(f"Loading files...\n1. {path1}\n2. {path2}")

    try:
        # 1. Load the Parquet files
        df1 = pd.read_parquet(path1)
        df2 = pd.read_parquet(path2)

        print(df1.head())
        print(df2.head())

        # 2. check if the exclusion column actually exists
        if exclude_col not in df1.columns or exclude_col not in df2.columns:
            print(
                f"⚠️ Warning: The column '{exclude_col}' was not found in one or both files."
            )
            # We continue anyway to check the rest of the dataframe

        # 3. Create views dropping the specific column
        # errors='ignore' allows the script to run even if 'fuel_kg' is missing
        df1_subset = df1.drop(columns=[exclude_col], errors="ignore")
        df2_subset = df2.drop(columns=[exclude_col], errors="ignore")

        # 4. Sort DataFrames (Optional but Recommended)
        # If your files aren't guaranteed to be in the exact same row order,
        # uncomment the lines below and replace 'ID_COLUMN' with your unique identifier.
        # df1_subset = df1_subset.sort_values(by='ID_COLUMN').reset_index(drop=True)
        # df2_subset = df2_subset.sort_values(by='ID_COLUMN').reset_index(drop=True)

        # 5. Compare the DataFrames
        print(f"\nComparing dataframes (ignoring '{exclude_col}')...")

        assert_frame_equal(df1_subset, df2_subset)

        print("✅ Success! The files are identical (excluding the fuel_kg column).")
        return True

    except AssertionError as e:
        print("❌ Mismatch found! The files differ.")
        print("-" * 30)
        print(e)  # This prints the detailed Pandas difference report
        print("-" * 30)
        return False

    except FileNotFoundError:
        print("❌ Error: One of the file paths provided does not exist.")
        return False
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")
        return False


# --- Usage ---
if __name__ == "__main__":
    # Replace these with your actual file paths
    file_a = "outspoken-tornado_v1.parquet"
    file_b = "outspoken-tornado_v1.parquet"
    compare_parquet_files(file_a, file_b)
