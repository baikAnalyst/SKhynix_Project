# 📈 논문 연구 및 국내기업 SK하이닉스에 적용하여 구현

## 📍 소개

토픽모델링을 통해 뉴스버즈량과 주식 거래량으로 거래량이 peak인 날짜의 주요 뉴스 토픽을 도출하여 금융 시장의 변동성을 설명하고자 함

## 📆 기간

2023.10.30~2023.11.03

## 📝 노션(WBS)

https://puddle-sandal-0bf.notion.site/fd17d595ed954d8aaea64ce0672a8fe3?v=2b353239cb4b4d3abf2840e774cbfebd&pvs=4 
 
## 👩‍💻 팀원

| 이름   | 역할                                         | url                          |
| ------ | -------------------------------------------- | ----------------------------- |
| 양소은 | 팀장, 논문 요약 리뷰 및 발표, 데이터 전처리(토큰화, 품사 부착, 불용어처리), LDA 토픽모델링, 내생변수 제거, FVE 계산, FPE 계산, 대시보드 제작(예정)           | https://github.com/Sunnn-y |
| 손효정 | 팀원, 뉴스데이터 수집, 5년치 데이터 병합, 병렬처리, 내생변수 제거, LASSO 회귀 분석, 토픽별 예측 거래량 시각화, FPE 계산, 최종 발표                         | https://github.com/sonhj110 |
| 백선영 | 팀원, 주식거래량데이터 수집, 피크데이 도출, LDA 시각화, mecab설치 및 전체 코드 실행, LDA 토픽모델링 지정한 토픽개수 출력 디버깅, 데이터파일(csv)가공 및 생성  | https://github.com/baikAnalyst |
| 신민채 | 팀원, 뉴스데이터와 주식거래량 시각화, 최종발표자료 제작       | https://github.com/shenmincai |

## 📚 데이터
| 데이터   | 참고                                     | 출처                          |
| ------ | -------------------------------------------- | ----------------------------- |
| 뉴스수집 | 네이버 증권 뉴스      | https://finance.naver.com/news/ |
| 주식거래량수집 | 네이버 증권 일별시세   | https://finance.naver.com/sise/ |


## 💻 개발환경

- Python(VSCode, Google Colab, Jupyter notebook)
- Numpy
- Pandas
- konlpy
- BeautifulSoup
- Scikit-learn
- Matplotlib
- Mecab
- gensim
- pyLDAvis

## 📝 구현내용

https://puddle-sandal-0bf.notion.site/fd17d595ed954d8aaea64ce0672a8fe3?v=2b353239cb4b4d3abf2840e774cbfebd
https://github.com/Sunnn-y/SKhynix_Project

## 📃 레퍼런스

[High quality topic extraction from business news explains abnormal financial market volatility]
Ryohei Hisano, Didier Sornette, Takayuki Mizuno, Takaaki Ohnishi, Tsutomu Watanabe




