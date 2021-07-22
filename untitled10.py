import re
import io
import os
import time
import warnings
import zipfile
from collections import defaultdict, namedtuple
from functools import reduce, wraps

import requests
import nltk
import scipy

import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import scale
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer

from surprise import Dataset, Reader, KNNWithMeans, SVD

from yellowbrick.cluster import KElbowVisualizer
from memory_profiler import memory_usage
from pytrends.request import TrendReq

# %%

rand_state = 10
samp_size = 5

sns.set_theme()

# %%

pytrend = TrendReq()
pytrend.build_payload(['online shopping'], timeframe='2020-01-01 2020-12-31')
df_trends = pytrend.interest_over_time()

ax = sns.lineplot(x=df_trends.index, y=df_trends['online shopping'])
ax.set(ylabel='iterest over time (%)', xlabel='')
ax.title.set_text("Google trend for search term 'online shopping'")

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

rng = np.random.RandomState(rand_state)

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

users_dists_mtx = (scipy.spatial.distance
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

# %%

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
(biased MAPE = {:.2f})""".format(custom_mape)

txt_3_1 = """Module Recommender:
est. rates for known
user-movie pairs
(biased MAPE = {:.2f})""".format(module_mape)

txt_1_2 = ''

txt_2_2 = """Custom Recommender:
est. rates for unknown
user-movie pairs
(to recommend or not)"""

txt_3_2 = """Module Recommender:
est. rates for unknown
user-movie pairs
(to recommend or not)"""


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

vmin, vmax = data_viz[0].agg(['min', 'max'])


def draw_heatmap(*args, **kwargs):

    d_raw = kwargs.pop('data')

    d_pvt = d_raw.pivot(index=args[0], columns=args[1], values=args[2])
    d_msk = (d_raw.pivot(index=args[0],
                         columns=args[1],
                         values=args[3])
             .astype(bool))

    sns.heatmap(d_pvt,
                annot=True,
                annot_kws={'fontsize': 13},
                fmt='.1f',
                vmin=vmin,
                vmax=vmax,
                cbar=False,
                mask=d_msk,
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

fg.set_titles(col_template='{col_name}',  size=15)
fg.fig.tight_layout()

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


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, cols):
        self.cols = cols
        self.ref_dic = None

    @staticmethod
    def _get_enc(ser):
        count = ser.value_counts()
        noise = abs(np.random.normal(5e-6, 1e-5, len(count)))
        return (count / count.sum() + noise).to_dict()

    def fit(self, X, y=None):
        self.ref_dic = {}
        for col in self.cols:
            col_enc = self._get_enc(X[col])
            self.ref_dic[col] = col_enc
        return self

    def transform(self, X, y=None):
        for col in self.cols:
            X[col] = X[col].apply(
                lambda x, col=col: self.ref_dic[col].get(x, 0))
        return X

# %%


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    return text


class StemmedTfidfVectorizer(TfidfVectorizer):

    english_stemmer = nltk.stem.SnowballStemmer('english')

    def build_analyzer(self):
        analyzer = TfidfVectorizer.build_analyzer(self)
        return lambda doc: (self.english_stemmer.stem(w)
                            for w in analyzer(doc))

# %%


data_pipe = make_pipeline(
    FrequencyEncoder(['year']),
    make_column_transformer(
        (StemmedTfidfVectorizer(stop_words='english',
                                preprocessor=preprocess_text,
                                decode_error='ignore',
                                max_features=500),
         'title_tags'),
        remainder='passthrough'),
    TruncatedSVD(n_components=50,
                 random_state=rand_state))

movs_data_enc = data_pipe.fit_transform(movs_data)

# %%

model = KMeans(random_state=rand_state)

visualizer = KElbowVisualizer(model,
                              k=10,
                              timings=False)
visualizer.fit(movs_data_enc)

visualizer.show()

# %%

model_tuned = KMeans(visualizer.elbow_value_, random_state=rand_state)

model_tuned.fit(movs_data_enc)

movs_data['cluster'] = model_tuned.labels_

movs_data.groupby('cluster').size()

cluster_dct = dict(zip(movs_data.index, model_tuned.labels_))

rats['cluster'] = rats['movieId'].map(lambda x: cluster_dct[x])

print(rats.groupby('cluster').size())

# %%

data = (Dataset
        .load_from_df(rats[['userId', 'movieId', 'rating']], Reader()))

raw_ratings = data.raw_ratings

kf = KFold(n_splits=3, shuffle=True, random_state=rand_state)

algo = SVD(random_state=rand_state)

# %%

# https://surprise.readthedocs.io/en/stable/FAQ.html#how-to-compute-precision-k-and-recall-k


def precision_recall_at_k(predictions, k=5, threshold=4):
    """Return precision and recall at k metrics for each user"""

    user_est_true = defaultdict(list)

    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    prec = dict()
    recs = dict()

    for uid, user_ratings in user_est_true.items():

        user_ratings.sort(key=lambda x: x[0], reverse=True)

        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)

        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])

        n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold))
                              for (est, true_r) in user_ratings[:k])

        prec[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0

        recs[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 0

    return prec, recs

# %%


res_metrics = []
Res = namedtuple('Res', ['fold', 'case', 'time', 'mem'])

# https://hakibenita.com/fast-load-data-python-postgresql


def profile(fn):

    @wraps(fn)
    def inner(*args, **kwargs):

        fold = kwargs.pop('fold')
        case = kwargs.pop('case')

        tm = time.perf_counter()
        retval = fn(*args, **kwargs)
        elapsed = time.perf_counter() - tm

        mem, retval = memory_usage(
            (fn, args, kwargs), retval=True, timeout=200, interval=1e-7)

        res_metrics.append(Res(fold, case, elapsed, max(mem) - min(mem)))
        return retval

    return inner

# %%


@profile
def get_predictions(trn_index, tst_index, alg=algo):

    A_raw_ratings = [raw_ratings[i] for i in trn_index]
    B_raw_ratings = [raw_ratings[i] for i in tst_index]

    data.raw_ratings = A_raw_ratings

    trainset = data.build_full_trainset()
    testset = data.construct_testset(B_raw_ratings)

    alg.fit(trainset)

    return alg.test(testset)


# %%

folds_metrics_data = []

for i, (train_index, test_index) in enumerate(kf.split(raw_ratings), 1):

    no_cl_predictions = get_predictions(train_index,
                                        test_index,
                                        fold=i,
                                        case='raw items')

    cl_predictions = []

    for cl in rats['cluster'].unique():

        cl_index = []
        for idx in [train_index, test_index]:
            cl_index.append(rats
                            .iloc[idx]
                            .pipe(lambda x, cl=cl:
                                  x[x['cluster'] == cl].index))

        cl_predictions.extend(get_predictions(*cl_index,
                                              fold=i,
                                              case='clustered'))

    for key, val in dict(zip(
            ['raw items', 'clustered'],
            [no_cl_predictions, cl_predictions])).items():

        precisions, recalls = precision_recall_at_k(val)

        folds_metrics_data.append(pd.concat(
            [pd.Series(precisions,
                       name="""SVD recommender's precision at k=5
(distribution and average over all users)"""),
             pd.Series(recalls,
                       name="""SVD recommender's recall at k=5
(distribution and average over all users)""")],
            axis=1)
            .assign(fold='fold #{}'.format(i))
            .assign(case=key))

# %%

folds_metrics_viz = (pd.concat(folds_metrics_data, ignore_index=True)
                     .melt(id_vars=['fold', 'case']))

row, col, hue = ('fold', 'variable', 'case')
means = (np.asarray(folds_metrics_viz
                    .groupby([row, col, hue])['value']
                    .mean())
         .reshape(-1, 2))

# %%

fg = sns.FacetGrid(data=folds_metrics_viz,
                   row=row,
                   col=col,
                   hue=hue,
                   margin_titles=True,
                   height=3,
                   aspect=2.25)

fg.map(sns.kdeplot,
       'value',
       clip=(0, 1),
       bw_adjust=1.25,
       linewidth=1.5)


for i, ax in enumerate(fg.axes.flat):

    if i == 0:
        ax.legend()

    # https://stackoverflow.com/questions/28956622/how-to-locate-the-median-in-a-seaborn-kde-plot

    for j, line in enumerate(ax.get_lines()):

        x, y = line.get_data()

        nearest_to_mean = np.abs(x - np.flip(means[i])[j]).argmin()
        x_mean = x[nearest_to_mean]
        y_mean = y[nearest_to_mean]

        ax.vlines(x=x_mean,
                  ymin=0,
                  ymax=y_mean,
                  linestyles='--',
                  colors=line.get_color(),
                  linewidth=1)

fg.set_titles(col_template='{col_name}',
              row_template='{row_name}',
              size=15)
fg.fig.tight_layout()

# %%

compare_res_metrics = (pd.DataFrame(res_metrics)
                       .groupby(['fold', 'case'])
                       .agg({'time': 'sum', 'mem': 'max'})
                       .groupby(level='case')
                       .mean())
