import CSVProcessor as cp
def main():
    try:
        file_path="titanic.csv"
        data=cp.load_file("titanic.csv")
        print("Data has loaded successfully.")
        columns=cp.total_columns(data)
        print(f"The number of columns are {columns}")
        rows=cp.total_rows(data)
        print(f"The number of rows are {rows}")
        filled_values=cp.fill_missing_val(data)
        print(f"The missing values are filled.")
        print(filled_values)
    except Exception as e:
        print(e)
if __name__=="__main__":
    main()