import pandas as pd
import numpy as np

df = pd.read_csv('/content/drive/MyDrive/sesac_jongro/SK하이닉스/tradingVolumeData/volume.csv', header=None, sep=',')
df.columns = ['Date', 'Volume']

# Remove unwanted data 
df = df.iloc[6:]
df.reset_index(inplace=True)
df.drop(labels='index', axis=1, inplace=True)

# to convert date string to 'YYYYMMDD' format
def convert_date(date_str):
    return ''.join(date_str.split('-'))

# to convert date values to int
df['Date'] = df['Date'].apply(convert_date)
df['Date'] = df['Date'].astype(int)


df = df.sort_values(by='Date', ascending=True) 

# Define the start and end dates for each 6-month bin
bin_dates = [
    (20181001, 20190331),  # Oct 2018 - Mar 2019
    (20190401, 20190930),  # Apr 2019 - Sep 2019
    (20191001, 20200331),  # Oct 2019 - Mar 2020
    (20200401, 20200930),  # Apr 2020 - Sep 2020
    (20201001, 20210331),  # Oct 2020 - Mar 2021
    (20210401, 20210930),  # Apr 2021 - Sep 2021
    (20211001, 20220331),  # Oct 2021 - Mar 2022
    (20220401, 20220930),  # Apr 2022 - Sep 2022
    (20221001, 20230331),  # Oct 2022 - Mar 2023
    (20230401, 20230930)   # Apr 2023 - Sep 2023
]

bin_labels = list(range(1, len(bin_dates) + 1))

# Create 6-month bins based on Date
df['6months_Bins'] = pd.cut(
    df['Date'],
    bins=[bin[0] for bin in bin_dates] + [bin_dates[-1][1]],  # Add the end date of the last bin
    labels=bin_labels,
    right=False
)


peak_days_df = pd.DataFrame()

# Find and print quantile values for each bin
for bin_label in bin_labels:
    bin_data = df[df['6months_Bins'] == bin_label]
    quantile_value = bin_data['Volume'].quantile(0.95)
    print(f'Bin {bin_label} Quantile Value: {quantile_value}')

    peak_days = bin_data[bin_data['Volume'] >= quantile_value]
    peak_days_df = pd.concat([peak_days_df, peak_days])

peak_days_df.reset_index(drop=True, inplace=True)

peak_days_df.to_csv('PeakDays.csv', index=False)
