import os
import pandas as pd
import inspect as insp
from sqlalchemy import create_engine
from IPython.display import display as original_display

# Database credentials
db_host = 'LBHHLWSQL0001.lbhf.gov.uk'
db_port = '1433'
db_name = 'IA_ODS'

# Create the connection string for SQL Server using pyodbc with Windows Authentication
connection_string = f'mssql+pyodbc://@{db_host}:{db_port}/{db_name}?driver=ODBC+Driver+17+for+SQL+Server&Trusted_Connection=yes'

# Create the database engine
engine = create_engine(connection_string)

# Function to clean labels in any plot functions
def clean_label(label):
    try:
        return label.replace('_', ' ').title()
    except AttributeError as e:
        print(f'Error cleaning label: {e}')
        return label
 
# Function for getting the name of a DataFrame
def get_var_name(var):
    try:
        for name, value in globals().items():
            if value is var:
                return name
    except Exception as e:
        print(f'Error getting variable name: {e}')
    return None
 
# Function to provide list for data sources as a DataFrame when conducting analysis
def header_list(df):
    try:
        df_list_ = df.copy()
        df_list = df_list_.columns.tolist()
        df_list = pd.DataFrame(df_list)
        new_header = df_list.iloc[0]  # Get the first row for the header
        df_list = df_list[1:]  # Take the data less the header row
        df_list.columns = new_header  # Set the header row as the df header
        df_list.reset_index(drop=True, inplace=True)  # Reset index
        return df_list
    except Exception as e:
        print(f'Error creating header list: {e}')
        return pd.DataFrame()
    
def read_directory(directory=False):
    if directory == False:
        directory = os.getcwd()
        
    files = os.listdir(directory)
    
    if directory == os.getcwd():
        print(f"Your Current Directory is: {directory}")
    else:
        print(f"Directory being read is: {directory}")

    print("Files in: %s\n" % (files))

# Function to provide list for data sources as a DataFrame when conducting analysis
def display(df):
    try:
        frame = insp.currentframe().f_back
        name = "Unnamed DataFrame"
        for var_name, var_value in frame.f_locals.items():
            if var_value is df:
                name = var_name
                break
        if name not in {'df', 'Unnamed DataFrame', 'unique_counts'}:
            print(f"DataFrame: {name}")
        original_display(df)
    except Exception as e:
        print(f'Error displaying DataFrame: {e}')

# Function to provide list for data sources as a DataFrame when conducting analysis
def unique_values(df, display_df=True):
    try:
        unique_values = {col: df[col].unique() for col in df.columns}
        max_length = max(len(values) for values in unique_values.values())
        unique_df_data = {}
        for col, values in unique_values.items():
            unique_df_data[col] = list(values) + [None] * (max_length - len(values))
        unique_df = pd.DataFrame(unique_df_data)
        if display_df:
            pd.set_option('display.max_rows', None)
            display(unique_df.head(100))
            pd.reset_option('display.max_rows')
        return unique_df
    except Exception as e:
        print(f'Error extracting unique values: {e}')
        return pd.DataFrame()

# Function to validate the data in a DataFrame
def validate_data(df, show_counts=True):
    try:
        # print 
        df_name = get_var_name(df)
        print(f'#########################################################################################################################################################################################\nDataFrame: {df_name}')
        
        # Snapshot the dataset
        display(df)
        
        # Check for unique values
        unique_counts = pd.DataFrame(df.nunique())
        unique_counts = unique_counts.reset_index().rename(columns={0:'No. of Unique Values', 'index':'Field Name'})
        print("Unique values per field:")
        pd.set_option('display.max_rows', None)
        display(unique_counts)
        pd.reset_option('display.max_rows')
        
        # Checking for duplicates
        duplicate_count = df.duplicated().sum()
        print("\nNumber of duplicate rows:")
        print(duplicate_count,'\n')
        info = df.info(show_counts=show_counts)
        display(info)
        # Summary stats
        print("\nSummary statistics:")
        display(df.describe())
        print('End of data validation\n#########################################################################################################################################################################################\n')
    except Exception as e:
        print(f'Error validating data: {e}')


# Function to provide list for data sources as a DataFrame when conducting analysis
def query_data(schema, data):
    try:
        # Define the SQL query
        query = f'SELECT * FROM [{schema}].[{data}]'
        # Load data into DataFrame
        df = pd.read_sql(query, engine)
        print(f'Successfully imported {data}')
        return df
    except Exception as e:
        print(f'Error querying data: {e}')
        return pd.DataFrame()


# Function to provide list for data sources as a DataFrame when conducting analysis
def export_to_csv(df, **kwargs):
    try:
        # Obtaining wanted directory
        directory = kwargs.get('directory',r"C:\Users\jf79\OneDrive - Office Shared Service\Documents\H&F Analysis\Python CSV Repositry")
        
        # Obtaining name of DataFrame
        df_name = kwargs.get('df_name',get_var_name(df))
        if not isinstance(df_name, str) or df_name == '_':
                df_name = input('Dataframe not found in global variables. Please enter a name for the DataFrame: ')

        file_path = f'{directory}\\{df_name}.csv'

        print(f'Exproting {df_name} to CSV...\n@ {file_path}\n')
        df.to_csv(file_path, index=False)
        print(f'Successfully exported {df_name} to CSV')
    except Exception as e:
        print(f'Error exporting to CSV: {e}')