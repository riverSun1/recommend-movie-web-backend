import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix
from implicit.als import AlternatingLeastSquares
import pickle

saved_model_fname="model/finalized_model.sav"
data_fname = "data/ratings.csv"
item_fname ="data/movies_final.csv"
weight = 10
#-------------------------------------------------------------------------
# model_train() 함수는 ratings 데이터를 이용해서 추천 엔진(=model)을 학습시키는 함수이다.
def model_train():
    ratings_df = pd.read_csv(data_fname)
    # userId와 movieId를 category 데이터 형태로 바꿔준다.
    ratings_df["userId"] = ratings_df["userId"].astype("category")
    ratings_df["movieId"] = ratings_df["movieId"].astype("category")

    # create a sparse matrix of all the users/repos
    # 어떤 유저가 어떤 영화에 얼마의 평점을 주었는 지를 행렬 형태로 표현해주는 함수인 coo_matrix함수를 사용한다.
    rating_matrix = coo_matrix(
        (
            ratings_df["rating"].astype(np.float32),
            (
                ratings_df["movieId"].cat.codes.copy(),
                ratings_df["userId"].cat.codes.copy(),
            ),
        )
    )
    # als_model을 생성하는 부분을 보면 factors, regularization, dtype, iteration의 변수를 조정할 수 있다.
    # factors: latent factor의 개수로 숫자가 클수록 기준의 개수가 많아지고 이는 다양한 사람들의 취향을 반영할 수 있다는 뜻이다
    # 단점으로는 오버피팅이라고하는 과적합이 발생할 가능성이 높아진다. 과적합이 일어날 경우 학습한 데이터에서는 아주 정확한 결과값이
    # 학습하지 않은 데이터에서는 좋지 않은 결과값이 나온다.
    # regularization: 이러한 과적합 문제를 방지하기 위한 변수로 숫자가 클수록 과적합을 막을 수 있으나 너무 큰 값을 넣을 경우에는
    # 추천의 정확도가 떨어질 확률이 높아진다.
    # dtype : rating의 데이터 형식이 float이기 때문에 np.float64를 사용한다.
    # iterations : 학습을 통해 parameter의 업데이트를 몇 번 할 것인지를 나타낸다. iteration의 횟수도 많을수록 과적합이 될 가능성이 높다
    # Collaborative Filtering 기반의 추천 시스템이므로 변수에 여러 값을 넣어서 실험해 볼 수 있다.
    als_model = AlternatingLeastSquares(
        factors=50, regularization=0.01, dtype=np.float64, iterations=50
    )

    als_model.fit(weight * rating_matrix)

    pickle.dump(als_model, open(saved_model_fname, "wb"))
    return als_model
#-------------------------------------------------------------------------
# 모델에 itemId를 입력하고, 가장 비슷한 11개의 영화를 결과로 반환하는 함수다.
# 11개로 한 이유는 자기 자신이 제일 높은 유사도로 나오기 때문에 첫 번째 결과를 제외한 10개의 결과를 얻기 위함이다.
# calculate_item_based 함수에서는 가장 비슷한 영화의 movieId만을 출력하기에 원하는 결과를 모두 표시하려면
# movieId, title, genres 등 movies_final.csv 파일에 있는 정보를 함께 반환해준다.
def calculate_item_based(item_id, items):
    loaded_model = pickle.load(open(saved_model_fname, "rb"))
    recs = loaded_model.similar_items(itemid=int(item_id), N=11)
    return [str(items[r]) for r in recs[0]]
#-------------------------------------------------------------------------
# item_based_recommendation은 원하는 데이터 형태를 모두 받을 수 있돌고 변환해주는 함수다.
def item_based_recommendation(item_id):
    ratings_df = pd.read_csv(data_fname)
    ratings_df["userId"] = ratings_df["userId"].astype("category")
    ratings_df["movieId"] = ratings_df["movieId"].astype("category")
    movies_df = pd.read_csv(item_fname)

    items = dict(enumerate(ratings_df["movieId"].cat.categories))
    try:
        parsed_id = ratings_df["movieId"].cat.categories.get_loc(int(item_id))
        # parsed_id는 기존 item_id와는 다른 모델에서 사용하는 id입니다.
        result = calculate_item_based(parsed_id, items)
    except KeyError as e:
        result = []
    result = [int(x) for x in result if x != item_id]
    result_items = movies_df[movies_df["movieId"].isin(
        result)].to_dict("records")
    return result_items
#-------------------------------------------------------------------------
# calculate_user_based에서는 recommend 함수를 사용하여 유저 기반의 추천을 한다.
# build_matrix_input 함수를 이용해서 user input을 정의해주어야 한다.
# input으로 받는 형태는 {1: 4.5, 2: 4.0}과 같이 python dictionary 형태로 받게 되는데
# recommend 함수에서 필요한 데이터 형태가 user-item 간의 coo_matrix이기 때문이다.
def calculate_user_based(user_items, items):
    loaded_model = pickle.load(open(saved_model_fname, "rb"))
    recs = loaded_model.recommend(
        userid=0, user_items=user_items, recalculate_user=True, N=10
    )
    return [str(items[r]) for r in recs[0]]
#-------------------------------------------------------------------------
def build_matrix_input(input_rating_dict, items):
    model = pickle.load(open(saved_model_fname, "rb"))
    # input rating list : {1: 4.0, 2: 3.5, 3: 5.0}

    item_ids = {r: i for i, r in items.items()}
    mapped_idx = [item_ids[s]
                  for s in input_rating_dict.keys() if s in item_ids]
    data = [weight * float(x) for x in input_rating_dict.values()]
    # print('mapped index', mapped_idx)
    # print('weight data', data)
    rows = [0 for _ in mapped_idx]
    shape = (1, model.item_factors.shape[0])
    return coo_matrix((data, (rows, mapped_idx)), shape=shape).tocsr()
#-------------------------------------------------------------------------
def user_based_recommendation(input_ratings):
    ratings_df = pd.read_csv(data_fname)
    ratings_df["userId"] = ratings_df["userId"].astype("category")
    ratings_df["movieId"] = ratings_df["movieId"].astype("category")
    movies_df = pd.read_csv(item_fname)

    items = dict(enumerate(ratings_df["movieId"].cat.categories))
    input_matrix = build_matrix_input(input_ratings, items)
    result = calculate_user_based(input_matrix, items)
    result = [int(x) for x in result]
    result_items = movies_df[movies_df["movieId"].isin(
        result)].to_dict("records")
    return result_items
#-------------------------------------------------------------------------
if __name__ == "__main__":
    model = model_train()