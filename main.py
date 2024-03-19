# 엔드포인트 관리

from typing import List, Optional
from fastapi import FastAPI, Query
from resolver import random_items, random_genres_items
from fastapi.middleware.cors import CORSMiddleware
from recommender import item_based_recommendation

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/all/")
async def all_movies():
    result = random_items()
    return {"result":result}
    # return {"message": "All movies"}

@app.get("/genres/{genre}")
async def genre_movies(genre:str):
    result = random_genres_items(genre)
    return {"result": result}
    # return {"message": f"genre: {genre}"}

@app.get("/user-based/")
async def user_based(params: Optional[List[str]] = Query(None)):
    return {"message": "user based"}

# /item-based 엔드포인트를 다음과 같이 item_based_recommendation 함수의 결과를 출력하도록 수정한다.
# /user-based 엔드포인트에서는 유저의 평점을 기반으로 한 추천 결과를 보여준다.
@app.get("/item-based/{item_id}")
async def item_based(item_id:str):
    result = item_based_recommendation(item_id)
    return {"result": result}
    # return {"message": f"item based: {item_id}"}