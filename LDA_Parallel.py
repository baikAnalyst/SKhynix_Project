import pandas as pd
import MeCab
from collections import Counter
from gensim import corpora
from gensim.models import LdaModel, TfidfModel
from joblib import Parallel, delayed # 병렬처리 모듈

df = pd.read_csv('sk_data.csv')

mecab = MeCab.Tagger()

# 대용량 데이터프레임을 읽거나 생성합니다.
big_dataframe = df

# 데이터프레임을 작은 블록으로 분할
block_size = 100
data_blocks = [df[i:i + block_size] for i in range(0, len(df), block_size)]

## 토큰화 및 품사 부착
df['morphs'] = None
df['pos'] = None

all_tokens = [] # 빈도수 확인할 전체 토큰 묶음 리스트

def get_tokens(block) :
  mecab = MeCab.Tagger()
  index = 0
    
  for i, row in block.iterrows():
    try:
      morphs = mecab.parse(row['2'])
      tokens = []
      pos_tag = []
      lines = morphs.split("\n")
      for line in lines:
          if "\t" in line: # 텍스트 있는 줄만 추출
              token, info = line.split("\t")
              token_info = info.split(",")
              surface = token # 토큰
              pos = token_info[0]  # 품사
              tokens.append(surface)
              pos_tag.append((surface, pos))
      df.at[i, 'morphs'] = ' '.join(tokens)
      df.at[i, 'pos'] = pos_tag
      all_tokens.extend(pos_tag)
    except:
      pass

  print(i,'번째 블록')


with Parallel(n_jobs=8) as parallel : # 병렬처리 실행
    results = parallel(delayed(get_tokens)(block) for block in data_blocks)

print(df.head())