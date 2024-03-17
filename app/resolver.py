# /all 엔드포인트

import pandas as pd

item_fname = 'data/movies_final.csv'

# pandas에서 제공되는 함수 중 sample이라는 함수를 이용해서 랜덤하게 10개의 데이터를 반환
# 함수가 호출될 때마다 랜덤한 10개의 영화 데이터가 반환
def random_items():
    movies_df = pd.read_csv(item_fname)
    movies_df = movies_df.fillna('') # 공백을 채워줍니다.
    result_items = movies_df.sample(n=10).to_dict("records")
    return result_items

# random_items 함수와 다르게 random_genres_items 함수는 genre를 input으로 받도록 만듭니다.
# 그래야 장르 이름이 들어왔을 때, 그에 따른 처리를 할 수 있게 됩니다.
def random_genres_items(genre):
    movies_df = pd.read_csv(item_fname)
    # input으로 받은 genre가 기존 movies_df에 있는 genres 컬럼 안에 포함되는지 안되는지를 판별합니다.
    genre_df = movies_df[movies_df['genres'].apply(lambda x: genre in x.lower())]
    genre_df = genre_df.fillna('') # 공백을 채워줍니다.
    result_items = genre_df.sample(n=10).to_dict("records")
    return result_items