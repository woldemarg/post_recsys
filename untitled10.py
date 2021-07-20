import warnings
import seaborn as sns
from sklearn.metrics import mean_absolute_percentage_error
from scipy.spatial import distance
from sklearn.preprocessing import scale
import numpy as np
from collections import defaultdict
from functools import reduce
import zipfile
import io
import os
import requests
from yellowbrick.cluster import KElbowVisualizer
from sklearn import cluster
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.model_selection import KFold
from surprise import SVD, Dataset, Reader, KNNWithMeans
# from surprise.model_selection import KFold
import pandas as pd
import bra_utils as butils

# %%

zip_path = 'https://files.grouplens.org/datasets/movielens/ml-latest-small.zip'
dir_path = 'movielens'

# %%

res = requests.get(zip_path)

with zipfile.ZipFile(io.BytesIO(res.content)) as arc:
    arc.extractall(dir_path)

# %%

movs = pd.read_csv(os.path.join(dir_path, 'ml-latest-small', 'movies.csv'))
rats = pd.read_csv(os.path.join(dir_path, 'ml-latest-small', 'ratings.csv'))
tags = pd.read_csv(os.path.join(dir_path, 'ml-latest-small', 'tags.csv'))

# %%

rng = np.random.RandomState(10)

samp_size = 5

rand_users = rng.choice(rats['userId'].unique(),
                        size=samp_size,
                        replace=False)

movs_by_rand_users = (rats[rats['userId'].isin(rand_users)]
                      .groupby('movieId')
                      .size()
                      .pipe(lambda x: x[x >= 3]))

rand_movs = rng.choice(movs_by_rand_users.index,
                       size=samp_size,
                       replace=False)

rats_tiny = (rats[(rats['userId'].isin(rand_users)) &
                  (rats['movieId'].isin(rand_movs))])

rats_tiny_mtx = rats_tiny.pivot(index='userId',
                                columns='movieId',
                                values='rating')

# %%

users_vecs_scaled = np.nan_to_num(
    scale(rats_tiny_mtx.to_numpy(),
          axis=1,
          with_std=False))

for i in np.where(~users_vecs_scaled.any(axis=1))[0]:

    users_vecs_scaled[i] = np.random.RandomState(i).normal(
        loc=0,
        scale=0.1,
        size=users_vecs_scaled.shape[1]) / np.iinfo(np.int64).max

users_dists_mtx = (distance
                   .cdist(users_vecs_scaled,
                          users_vecs_scaled,
                          'cosine'))

users_dists_mtx_no_diagonal = (users_dists_mtx[~np.eye(
    users_dists_mtx.shape[0], dtype=bool)]
    .reshape(users_dists_mtx.shape[0], -1))

rats_tiny_arr = rats_tiny_mtx.to_numpy()


def to_weights(dist_vec):
    rv = [1 / np.finfo(np.float64).eps if x == 0 else 1 / x for x in dist_vec]
    sm = sum(rv)
    return np.array([x / sm for x in rv])


rates_est = np.array([])

for i in range(len(rats_tiny_arr)):

    rats_tiny_arr_no_cur_user = np.delete(rats_tiny_arr, i, axis=0)

    for cur_mov_rates_by_knn in rats_tiny_arr_no_cur_user.T:

        notna_idx = set(np.argwhere(~np.isnan(cur_mov_rates_by_knn)).flat)

        cur_user_dists_to_knn = users_dists_mtx_no_diagonal[i]

        notneg_idx = set(np.argwhere(cur_user_dists_to_knn > 0).flat)

        knn_idx = list(notna_idx.intersection(notneg_idx))

        user_mov_rate_est = np.dot(
            cur_mov_rates_by_knn[knn_idx],
            to_weights(cur_user_dists_to_knn[knn_idx]))

        rates_est = np.append(rates_est, user_mov_rate_est)

rates_test_idx = np.argwhere(~np.isnan(rats_tiny_arr.flat))

custom_mape = mean_absolute_percentage_error(
    rats_tiny_arr.flat[rates_test_idx].ravel(),
    rates_est[rates_test_idx])


# %%

tiny_data = (Dataset
             .load_from_df(
                 rats_tiny[['userId', 'movieId', 'rating']],
                 Reader(rating_scale=(1, 5))))

tiny_trainset = tiny_data.build_full_trainset()
tiny_testset = tiny_trainset.build_testset()
tiny_anti_testset = tiny_trainset.build_anti_testset()

algo = KNNWithMeans(sim_options={'name': 'cosine'}, verbose=False)
algo.fit(tiny_trainset)

tiny_estimations = algo.test(tiny_testset)

all_users_est_true = defaultdict(list)

for userId, movId, r_true, r_est, _ in tiny_estimations:
    all_users_est_true[(userId, movId)].extend((r_true, r_est))

r_true, r_est = zip(*all_users_est_true)

module_mape = mean_absolute_percentage_error(*zip(*list(
    all_users_est_true.values())))


# %%

mlt_idx = pd.MultiIndex.from_product(
    [rats_tiny_mtx.index,
     rats_tiny_mtx.columns])

custom_rec = pd.DataFrame(rates_est,
                          index=mlt_idx)

# %%

tiny_predictions = algo.test(tiny_anti_testset)

for userId, movId, r_true, r_est, _ in tiny_predictions:
    all_users_est_true[(userId, movId)].extend((r_true, r_est))

module_rec = pd.Series({k: v[1]
                        for k, v in all_users_est_true.items()}).to_frame()

module_rec = module_rec.reindex(mlt_idx)

# %%

orig_mtx = rats_tiny_mtx.copy().T.unstack().to_frame()

# %%

txt_1_1 = """Initial matrix:
true rates for known
user-movie pairs
(to find similar users)"""

txt_2_1 = """Custom Recommender:
est. rates for known
user-movie pairs
(MAPE = {:.2f})""".format(custom_mape)

txt_3_1 = """Module Recommender:
est. rates for known
user-movie pairs
(MAPE = {:.2f})""".format(module_mape)

txt_1_2 = ''

txt_2_2 = """Custom Recommender:
est. rates for unknown
user-movie pairs
(to recommend)"""

txt_3_2 = """Module Recommender:
est. rates for unknown
user-movie pairs
(to recommend)"""


orig_mtx['type'] = txt_1_1
orig_mtx['mask'] = 1
orig_mtx.loc[orig_mtx[0].notna(), 'mask'] = 0

custom_rec['type'] = txt_2_1
custom_rec['mask'] = 1
custom_rec.iloc[rates_test_idx.flat,
                custom_rec.columns.get_loc('mask')] = 0

module_rec['type'] = txt_3_1
module_rec['mask'] = 1
module_rec.iloc[rates_test_idx.flat,
                module_rec.columns.get_loc('mask')] = 0

data_viz = pd.concat([orig_mtx, custom_rec,  module_rec]).reset_index()

data_viz_copy = data_viz.copy()

data_viz_copy['type'] = data_viz_copy['type'].replace(
    {txt_1_1: txt_1_2,
     txt_2_1: txt_2_2,
     txt_3_1: txt_3_2})

data_viz_copy['mask'] = abs(data_viz_copy['mask'] - 1)

data_viz = pd.concat([data_viz, data_viz_copy])

# %%

sns.set_theme()


def draw_heatmap(*args, **kwargs):

    data = kwargs.pop('data')

    d = data.pivot(index=args[0], columns=args[1], values=args[2])
    d_mask = (data.pivot(index=args[0],
                         columns=args[1],
                         values=args[3])
              .astype(bool))

    sns.heatmap(d,
                annot=True,
                annot_kws={'fontsize': 13},
                fmt='.1f',
                cbar=False,
                mask=d_mask,
                **kwargs)


fg = sns.FacetGrid(data_viz,
                   col='type',
                   col_wrap=3,
                   height=4.75,
                   aspect=0.8)

# https://docs.python.org/3/library/warnings.html#temporarily-suppressing-warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fg.map_dataframe(draw_heatmap,
                     'userId',
                     'movieId',
                     0,
                     'mask',
                     cmap="Blues",
                     square=True,
                     linewidth=0.1)

for i, ax in enumerate(fg.axes.flat):
    ax.set_aspect('equal', 'box')
    ax.set_xlabel('movies')
    if i in [0, 3]:
        ax.set_ylabel('users')
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=10)

fg.set_titles(col_template="{col_name}",  size=15)
fg.fig.tight_layout()

# %%


def precision_recall_at_k(predictions, k=5, threshold=3.5):
    """Return precision and recall at k metrics for each user"""

    # First map the predictions to each user.
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions = dict()
    recalls = dict()
    for uid, user_ratings in user_est_true.items():

        uid = 1
        user_ratings = user_est_true[uid]
        y_score, y_true = zip(*user_ratings)
        y_true_2 = [true_r >= threshold for true_r in y_true]
        y_score_2 = [est_r >= threshold for est_r in y_score]

        top_k_accuracy_score(y_true_2, y_score_2, k=5)

        from sklearn.metrics import top_k_accuracy_score

        # Sort user ratings by estimated value
        user_ratings.sort(key=lambda x: x[0], reverse=True)

        # Number of relevant items
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)

        # Number of recommended items in top k
        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])

        # Number of relevant and recommended items in top k
        n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold))
                              for (est, true_r) in user_ratings[:k])

        # Precision@K: Proportion of recommended items that are relevant
        # When n_rec_k is 0, Precision is undefined. We here set it to 0.

        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0

        # Recall@K: Proportion of relevant items that are recommended
        # When n_rel is 0, Recall is undefined. We here set it to 0.

        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 0

    return precisions, recalls


# %%

movs_genre_dct = {}
for idx, row in movs.iterrows():
    movs_genre_dct[row['movieId']] = row['genres'].split('|')


movs_genre_dfr = (pd.DataFrame
                  .from_dict(movs_genre_dct, orient='index')
                  .stack()
                  .to_frame()
                  .reset_index()
                  .drop('level_1', axis=1)
                  .rename(columns={'level_0': 'movieId',
                                   0: 'genre'})
                  .assign(values=1)
                  .pivot_table(values='values',
                               index='movieId',
                               columns='genre',
                               fill_value=0,
                               dropna=False)
                  .reset_index())

# %%

movs['year'] = movs['title'].str.extract(r'\(.*(\d{4})\)')

# %%

tags_dct = defaultdict(set)
for idx, row in tags.iterrows():
    tags_dct[row['movieId']].add(row['tag'])

tags_dct = {k: ' '.join(v) for k, v in tags_dct.items()}

tags_dfr = (pd.DataFrame
            .from_dict(tags_dct, orient='index')
            .reset_index()
            .rename(columns={'index': 'movieId',
                             0: 'tags'}))

# %%


movs_data = reduce(lambda left, right:
                   pd.merge(left,
                            right,
                            how='outer',
                            on='movieId',
                            copy=False),
                   [movs, movs_genre_dfr, tags_dfr])


movs_data['title_tags'] = movs_data['title'].str.cat(movs_data['tags'],
                                                     sep=' ',
                                                     na_rep='')

# %%

movs_data = (movs_data
             .drop(['title',
                    'genres',
                   'tags'],
                   axis=1)
             .set_index('movieId'))

movs_data.fillna(movs_data.mode().iloc[0], inplace=True)

# %%

data_pipe = make_pipeline(
    butils.FrequencyEncoder(['year']),
    make_column_transformer(
        (butils.StemmedTfidfVectorizer(stop_words='english',
                                       preprocessor=butils.preprocess_text,
                                       decode_error='ignore',
                                       max_features=500),
         'title_tags'),
        remainder='passthrough'),
    TruncatedSVD(n_components=50,
                 random_state=1234))

movs_data_enc = data_pipe.fit_transform(movs_data)

# %%

model = cluster.KMeans(random_state=1234)

visualizer = KElbowVisualizer(model,
                              k=10,
                              timings=False)
visualizer.fit(movs_data_enc)

visualizer.show()

# %%

kmeans = cluster.KMeans(visualizer.elbow_value_, random_state=1234)
kmeans.fit(movs_data_enc)

movs_data['cluster'] = kmeans.labels_

movs_data.groupby('cluster').size()

cluster_dct = dict(zip(movs_data.index, kmeans.labels_))

rats['cluster'] = rats['movieId'].map(lambda x: cluster_dct[x])


# %%

data = (Dataset
        .load_from_df(rats[['userId', 'movieId', 'rating']], Reader()))

raw_ratings = data.raw_ratings

# kf = KFold(n_splits=5, random_state=1234)  # surprise
kf = KFold(n_splits=5, shuffle=True, random_state=1234)  # sklearn

algo = SVD(random_state=1234)

# %%

# for trainset, testset in kf.split(data):
#     algo.fit(trainset)
#     predictions = algo.test(testset)
#     precisions, recalls = precision_recall_at_k(predictions, k=5, threshold=4)

#     # Precision and recall can then be averaged over all users
#     print(sum(prec for prec in precisions.values()) / len(precisions))
#     print(sum(rec for rec in recalls.values()) / len(recalls))

# %%


def get_predictions(train_index, test_index, algo=algo):
    A_raw_ratings = [raw_ratings[i] for i in train_index]
    B_raw_ratings = [raw_ratings[i] for i in test_index]

    data.raw_ratings = A_raw_ratings

    trainset = data.build_full_trainset()
    testset = data.construct_testset(B_raw_ratings)

    algo.fit(trainset)

    return algo.test(testset)

# %%


for train_index, test_index in kf.split(raw_ratings):
    predictions = get_predictions(train_index, test_index)
    precisions, recalls = precision_recall_at_k(predictions)

    # Precision and recall can then be averaged over all users
    print(sum(prec for prec in precisions.values()) / len(precisions))
    print(sum(rec for rec in recalls.values()) / len(recalls))

# %%

for train_index, test_index in kf.split(raw_ratings):

    cl_predictions = []

    for cl in rats['cluster'].unique():

        cl_index = []
        for idx in [train_index, test_index]:
            cl_index.append(rats
                            .iloc[idx]
                            .pipe(lambda x, cl=cl:
                                  x[x['cluster'] == cl].index))

        cl_predictions.extend(get_predictions(*cl_index))

    precisions, recalls = precision_recall_at_k(cl_predictions)

    # Precision and recall can then be averaged over all users
    print(sum(prec for prec in precisions.values()) / len(precisions))
    print(sum(rec for rec in recalls.values()) / len(recalls))

# %%
