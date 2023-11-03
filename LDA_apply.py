import pandas as pd
import MeCab
from collections import Counter
from gensim import corpora
from gensim.models import LdaModel, TfidfModel
from pandas import DataFrame
# import pyLDAvis
# import pyLDAvis.gensim_models as gensimvis

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
stop_word = ['이날','보다','거래','만원','포인트','종가','지수','세대','어치','개인','대하','이어',
             '뉴시스','이틀','전일','거대','박수민','울엄마','스터리','간벌','헤럴드','항목','약지',
             '%↓','-',"\'…",'지디','투데이','기자','학생','그림','학과','동년배','%)','%,','com',
             '세에','chosun','조선일보','에게선','오토캠핑','전월','연합뉴스','대다수','대중','kr',
             '합니다','보였','십니까','참기름','까마득','문순','일지','고려대','오르내리','조기','간밤',
             '강세','식목일','데이터','학년도','코스닥', '코스피', '하락', '매수', '순매도', '거래일',
             '금리', '외국인', '마감', '상승', '종목', '기관', '분기','기업','차량','데일리안','만기',
             '미스트','경제','작성','구성원','가정용품','경기도','가격','정오뉴스','금요일','앵커',
             '어제','그룹','내년','시스','단위','기사','고르기','달러','종류','중구','당사','팟캐스트',
             '대학','명동','상위','사장','이사','계감','초당','이익','동기','데일리','휴가','순위',
             '스포츠서울','분야','생산','한경','복장','검색','대차','열흘','야구','반면','구역','지난해',
             '전년','어르신','온제','근처','등급','진작','무인','기사문']

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

n = 50 # 토픽개수
lda = LdaModel(corpus=corpus_TFIDF,
               id2word=id2word,
               num_topics=n,
               random_state=100)

# for t in lda.print_topics(num_topics=n):
#   print(t)
  

# for i, topic_list in enumerate(lda[corpus_TFIDF]):
#     if i==5:
#         break
#     print(i,'번째 문서의 topic 비율은',topic_list)



def make_topictable_per_doc(lda, corpus):
    topic_table = pd.DataFrame(columns=['문서 번호', '가장 비중이 높은 토픽', '가장 높은 토픽의 비중', '각 토픽의 비중'])

    for i, topic_list in enumerate(lda[corpus_TFIDF]):
        doc = topic_list[0] if lda.per_word_topics else topic_list            
        doc = sorted(doc, key=lambda x: (x[1]), reverse=True) # 비중이 높은 순으로 토픽 정렬

        for j, (topic_num, prop_topic) in enumerate(doc): #  몇 번 토픽인지와 비중을 나눠서 저장
            if j == 0:  # 가장 비중이 높은 토픽
                doc_num = i  # 문서 번호
                most_topic = int(topic_num)
                pro_topic = round(prop_topic, 4)
                topic_lis = topic_list
                new_row = pd.Series({'문서 번호': doc_num, '가장 비중이 높은 토픽': most_topic, '가장 높은 토픽의 비중': pro_topic, '각 토픽의 비중': topic_lis})
                topic_table = pd.concat([topic_table, new_row.to_frame().T], ignore_index=True)
            else:
                break

    return topic_table

topictable = make_topictable_per_doc(lda, corpus_TFIDF)
print(topictable)
