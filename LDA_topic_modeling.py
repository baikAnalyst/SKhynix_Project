import pandas as pd
import MeCab
from collections import Counter
from gensim import corpora
from gensim.models import LdaModel, TfidfModel

df = pd.read_csv('sk_data.csv')

mecab = MeCab.Tagger()


## 토큰화 및 품사 부착
df['morphs'] = None
df['pos'] = None

all_tokens = [] # 빈도수 확인할 전체 토큰 묶음 리스트
for i, row in df.iterrows():
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


## 각 토큰별로 문서 빈도수 확인
# print(Counter(all_tokens).most_common())


## 불용어 처리
stop_pos = ['NR','JKC','JKS','JKG','JKO','JKB','JKV','JKQ','JX','EP','EF','EC','ETN','ETM','SF','SE','SS','SP','SO','SW','SN','MM','NA','MAG','MAJ']
N_pos = ['NNG','NNP']
stop_word = ['이날','보다','거래','만원','포인트','종가','지수','세대','어치','개인','대하','이어','뉴시스','이틀','전일','거대','박수민','울엄마','스터리','간벌','헤럴드','항목','약지','%↓','-',"\'…",'지디','투데이','기자','학생','그림','학과','동년배','%)','%,','com','세에','chosun','조선일보','에게선','오토캠핑','전월','연합뉴스','대다수','대중','kr','합니다','보였','십니까','참기름','까마득','문순','일지','고려대','오르내리','조기','간밤','강세','식목일','데이터','학년도']

def preprocess(text):
    filtered_text = []
    for word, pos in text:
        if len(word) >= 2 and word not in stop_word and pos in N_pos:
            filtered_text.append((word, pos))
    return filtered_text


## 불용어 처리한 토큰리스트 생성
def make_tokens(df):
    df['tokens'] = None
    for i, row in df.iterrows():
      try:
        token = preprocess(df['pos'][i])
        df.at[i, 'tokens'] = [word for word, pos in token]  # 불용어 처리한 토큰만 남김
      except:
        pass
    return df

df = make_tokens(df)


## LDA
tokenized_docs = df['tokens'].dropna().apply(lambda x: ' '.join(x)).apply(lambda x: x.split())
id2word = corpora.Dictionary(tokenized_docs)
corpus_TDM = [id2word.doc2bow(doc) for doc in tokenized_docs]
tfidf = TfidfModel(corpus_TDM)
corpus_TFIDF = tfidf[corpus_TDM]

n = 20 # 토픽개수
lda = LdaModel(corpus=corpus_TFIDF,
               id2word=id2word,
               num_topics=n,
               random_state=100)

for t in lda.print_topics(num_topics=n):
  print(t)

# print(df.head())
