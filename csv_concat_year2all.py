import pandas as pd
import csv


total = pd.DataFrame()

for year in range(18,24) :

  temp = pd.read_csv(f'/content/drive/MyDrive/SK하이닉스/뉴스 크롤링 데이터/sk_{year}.csv', header = None, sep=',', encoding='utf-8')
  temp = temp[1:]
  temp.drop([0], axis=1, inplace=True)
  total = pd.concat([total, temp])

total.to_csv(f"/content/drive/MyDrive/SK하이닉스/뉴스 크롤링 데이터/sk_data.csv")