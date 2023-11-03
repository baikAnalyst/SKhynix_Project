import pandas as pd
from glob import glob

# 연도, 월 입력
year = 18
month = 12

# 폴더위치
loc = f'{year}.{month}월'

# 파일이름
if month >= 10 :
  file_name = f'sk{year}{month}'
else :
  file_name = f'sk{year}0{month}'

articles = glob(f"/content/drive/MyDrive/SK하이닉스/뉴스 크롤링 데이터/{loc}/*.csv") # 폴더 내의 모든 csv파일 목록을 불러온다
total_articles = pd.DataFrame() # 빈 데이터프레임 하나를 생성한다

for article in articles:
    temp = pd.read_csv(article, header = None, sep=',', encoding='utf-8') # csv파일을 하나씩 열어 임시 데이터프레임으로 생성한다
    total_articles = pd.concat([total_articles, temp]) # 전체 데이터프레임에 추가하여 넣는다

total_articles.to_csv(f"/content/drive/MyDrive/SK하이닉스/뉴스 크롤링 데이터/{loc}/{file_name}.csv") # 저장