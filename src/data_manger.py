import pandas as pd

df = pd.read_csv(r'C:\Users\user\PycharmProjects\Syntropy-Labs\src\testing data\clean_final_data.csv')

def auto_cleaning_df(dataframe=df, target_column=None):
    dataframe = dataframe.copy()

    target_column = None

    dataframe = dataframe.drop_duplicates()
    dataframe = dataframe.select_dtypes(include=['float64'])

    if target_column and target_column in dataframe:
        target_copy = dataframe[target_column]
        dataframe = dataframe.drop(columns=[target_column])


    for col in dataframe.columns:
        if dataframe[col].nunique(dropna=True) == len(dataframe):
            dataframe = dataframe.drop(columns=[col])

    #for uniq_col_value in dataframe.columns:
    #    if dataframe[uniq_col_value] == dataframe.columns.values:
    #        dataframe = dataframe.drop(columns=dataframe[uniq_col_value])
    #        print(dataframe[uniq_col_value])

    return dataframe

print(auto_cleaning_df(dataframe=df, target_column=None))