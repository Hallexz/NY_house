import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def data_process(file_name):
    df = pd.read_csv(file_name)
    df = df.fillna(value=0)
    df = df.drop_duplicates()
    df = df.drop(columns=['BROKERTITLE', 'ADMINISTRATIVE_AREA_LEVEL_2',
                          'STATE', 'MAIN_ADDRESS', 'FORMATTED_ADDRESS',
                          'LOCALITY', 'SUBLOCALITY', 'TYPE'])
    df[['PRICE', 'BEDS']] = df[['PRICE', 'BEDS']].astype('float64')
    return df

house = data_process('NY_House_Dataset.csv')

house.hist(bins=50, figsize=(20, 15))
plt.show()

plt.boxplot(x=house['PRICE'])
plt.show()

lower_percentile = house['PRICE'].quantile(0.01)
upper_percentile = house['PRICE'].quantile(0.95)

house = house[(house['PRICE'] > lower_percentile) & (house['PRICE'] < upper_percentile)]

plt.boxplot(x=house['PRICE'])
plt.show()


numeric_columns = house.loc[:, house.dtypes != object].columns
house[numeric_columns].describe()

num_features_corr = house[numeric_columns].corr()
num_features_corr
plt.figure(figsize=(12, 12))
sns.heatmap(num_features_corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('CORRELATION HEATMAP')
plt.show()

sns.pairplot(house, vars=['STREET_NAME', 'BEDS', 'BATH', 'ADDRESS'])
plt.show()

