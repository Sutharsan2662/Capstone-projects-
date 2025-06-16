import pandas as pd
# Specify the file path

#forest data

file_path = "Original_forest_data.xlsx"

# Read the Excel file with multiple sheets
excel_data = pd.ExcelFile(file_path)

# Get all sheet names
sheet_names = excel_data.sheet_names

# Read data from all sheets into a dictionary
sheets_dict = {sheet: excel_data.parse(sheet) for sheet in sheet_names}

# Example: Convert sheets_dict to a single DataFrame; this will add a new sheet column at the last.
combined_df_forest = pd.concat(
    [df.assign(Sheet=sheet_name) for sheet_name, df in sheets_dict.items()],
    ignore_index=True)
	
# later u can Drop the 'Sheet' column
combined_df_forest = combined_df.drop(columns=['Sheet'])

combined_df_forest.to_csv("combined_forest_corr.csv", index = False)


# grassland data
file_path = "Original_grassland_data.xlsx"

# Read the Excel file with multiple sheets
excel_data = pd.ExcelFile(file_path)

# Get all sheet names
sheet_names = excel_data.sheet_names

# Read data from all sheets into a dictionary
sheets_dict = {sheet: excel_data.parse(sheet) for sheet in sheet_names}

# Example: Convert sheets_dict to a single DataFrame; this will add a new sheet column at the last.
combined_df_grassland = pd.concat(
    [df.assign(Sheet=sheet_name) for sheet_name, df in sheets_dict.items()],
    ignore_index=True)
	
# later u can Drop the 'Sheet' column
combined_df_grassland = combined_df.drop(columns=['Sheet'])

combined_df_grassland.to_csv("combined_forest_corr.csv", index = False)