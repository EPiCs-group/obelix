import pandas as pd
import re
import matplotlib.pyplot as plt

def sort_dft_data(input_file, write_DFT = False, output_csv = None):
    df = pd.read_csv(input_file)

    #insert empty first row, then put the conformer number there (extracting from filename)
    df.insert(0, 'Conformer number', '')

    for index, row in df.iterrows():
        number = re.search(r"_([0-9]+)\.", row[1]).group(1)
        df.at[index, 'Conformer number'] = number


    #sort data based on the conformer numbers: 
    df['Conformer number'] = pd.to_numeric(df['Conformer number'])
    df_sorted = df.sort_values(by='Conformer number')
    df_sorted = df_sorted.drop(columns=df.columns[0])

    if write_DFT: 
       df_sorted.to_csv(output_csv, index=False) 
    
    return df_sorted


def crest_data (input_file):
    crest_df = pd.read_csv(input_file)
    return crest_df



def plotting(dft_df, crest_df):
    y = crest_df.iloc[:, 0]
    x = dft_df.iloc[:, 3]
    for i in range(len(x)):
        plt.scatter(x[i], y[i], color='darkcyan')  
        plt.text(x[i], y[i], str(i+1), ha='center', va='bottom', fontsize=10)

    plt.xlabel('DFT energy')
    plt.ylabel('GFN2-xTB energy')
    plt.grid()
    plt.show()

crest_df = crest_data(f"C:\\Users\\finta\\Documents\\tu_delft\\MEP\\CREST\\Absolute_energies\\abs_en_ceL191.csv")
dft_df = sort_dft_data(f"C:\\Users\\finta\\Documents\\tu_delft\\MEP\\DFT_input\\DFT_output\\ce191\\info_ceL191.csv")
plotting(dft_df, crest_df)