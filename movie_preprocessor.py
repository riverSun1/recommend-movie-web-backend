import pandas as pd
import requests
from tqdm import tqdm
import os
import time

def add_url(row):
    return f"http://www.imdb.com/title/tt{row}/"

def add_rating(df):
    ratings_df = pd.read_csv('data/ratings.csv')
    ratings_df['movieId'] = ratings_df['movieId'].astype(str)
    # agg : 집계, groupby : 기준이 되는 열 이름
    agg_df = ratings_df.groupby('movieId').agg(
        rating_count=('rating', 'count'),
        rating_avg=('rating', 'mean')
    ).reset_index()

    rating_added_df = df.merge(agg_df, on='movieId')
    return rating_added_df

def add_poster(df):
    API_KEY = os.environ.get('TMDB_API_KEY')
    
    for i, row in tqdm(df.iterrows(), total=df.shape[0]):
        tmdb_id = row["tmdbId"]
        # 발급받은 API 키와 tmdbId를 조합해서 tmdb_url을 생성한다.
        tmdb_url = f"https://api.themoviebd.org/3/movie/{tmdb_id}?api_key={API_KEY}&language=en-US"
        # tmdb_url로 HTTP GET Request 요청을 보낸 결과를 result에 저장한다.
        result = requests.get(tmdb_url)
        # final url : https://image.tmdb.org/t/p/original/uXDfjJbdP4ijW5hWSBrPrlKpxab.jpg

        try:
            # poster 경로는 result.json()을 https://image.tmdb.org/t/p/original/ 뒤에 붙인 url이 된다.
            # 이 경로에서 poster_path를 받아온다. 포스터 경로
            # 메인 함수에서 poster_path 컬럼을 None으로 선언한 후 add_poster함수에서 api를 통해서 poster_path 값을 받아서 컬럼 값에 저장한다.
            df.at[i, "poster_path"] = "https://image.tmdb.org/t/p/original" + \
                result.json['poster_path']
            time.sleep(0.1) # 0.1초 시간 간격

            # 토이스토리 포스터는 디폴트 값
        except (TypeError, KeyError) as e:
            df.at[i, "poster_path"] = "https://image.tmdb.org/t/p/original/uXDfjJbdP4ijW5hWSBrPrlKpxab.jpg"
    return df
        
if __name__ == "__main__":
    movies_df = pd.read_csv('data/movies.csv')
    movies_df['movieId'] = movies_df['movieId'].astype(str)
    links_df = pd.read_csv('data/links.csv', dtype=str)
    merged_df = movies_df.merge(links_df, on='movieId', how='left')
    merged_df['url'] = merged_df['imdbId'].apply(lambda x: add_url(x))
    # print(merged_df)
    ## print(movies_df.columns)
    ## print(links_df.columns)
    result_df = add_rating(merged_df)
    # print(result_df)
    result_df['poster_path'] = None
    result_df = add_poster(result_df)

    # 최종적으로 만들어진 테이블을 data밑의 movies_final.csv 이름의 csv 형식의 파일로 저장한다.
    result_df.to_csv("data/movies_final.csv", index=None)