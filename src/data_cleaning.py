import pandas as pd

def xlsx_to_csv():
    """Convert all XLSX files to CSV in the processed folder"""
    
    # the 2025 data was already in CSV format, so no need to process it
    for year in range(2018, 2025):
        xlsx_path = f"data/raw/{year}.xlsx"
        csv_path = f"data/processed/{year}.csv"
        
        df = pd.read_excel(xlsx_path)
        df.to_csv(csv_path, index=False, encoding="utf-8")
        print(f"✓ Saved to {csv_path} ({len(df)} rows)")

def merge_all_csv():
    """Merges all CSV files into one CSV, in the final folder"""

    df_list = []

    for year in range(2018, 2026):
        df = pd.read_csv(f"data/processed/{year}.csv")
        df_list.append(df)

    merged = pd.concat(df_list, ignore_index=True)
    merged.to_csv(f"data/final/final.csv", index=False, encoding="utf-8")
    print(f"✓ Merged {len(merged)} total rows into data/final/merged_data.csv")

def main():
    xlsx_to_csv()
    merge_all_csv()

if __name__ == "__main__":
    main()