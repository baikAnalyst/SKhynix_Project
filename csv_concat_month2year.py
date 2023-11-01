import pandas as pd
import csv

# 연도 입력
year = 20

total = pd.DataFrame()

for month in range(1,13) :

  if year == 23 and month >= 10 :
    break

  if year == 18 and month <= 9 :
    continue

  # 폴더위치
  loc = f'{year}.{month}월'

  # 파일이름
  if month >= 10 :
    file_name = f'sk{year}{month}'
  else :
    file_name = f'sk{year}0{month}'

  temp = pd.read_csv(f'/content/drive/MyDrive/SK하이닉스/뉴스 크롤링 데이터/{loc}/{file_name}.csv', header = None, sep=',', encoding='utf-8')
  temp = temp[1:]
  temp.drop([0,1,3,5], axis=1, inplace=True)
  total = pd.concat([total, temp])

total.to_csv(f"/content/drive/MyDrive/SK하이닉스/뉴스 크롤링 데이터/sk_{year}.csv")