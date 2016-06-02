import libpmf
import argparse
import os
import numpy as np
import logging
from sklearn import metrics, decomposition
from itertools import product
from pymaptools.benchmark import PMTimer
from pymaptools.unionfind import UnionFind
from pymaptools.sparse import dd2coo
from pymaptools.iter import isiterable
from collections import defaultdict, namedtuple
from functools import partial
from sklearn.cross_validation import train_test_split, KFold


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


Rating = namedtuple('Rating', 'user product rating'.split())

pmf_svd_grid = {
    'N': [0],
    'k': [2, 4, 8],
    'l': [0.1, 1.0, 10.0, 100.0]
}

pmf_nmf_grid = {
    'N': [1],
    'k': [2, 4, 8],
    'l': [0.1, 1.0, 10.0, 100.0]
}

skl_nmf_grid = {
    'n_components': [8, 16],
    'max_iter': [50, 100],
    'alpha': [3000.0, 10000.0, 30000.0],
    'solver': 'cd',
    'l1_ratio': [0.0, 0.1, 1.0],
    'tol': [1e-5, 1e-6],
    'init': ['nndsvd']
}

skl_svd_grid = {
    'n_components': [4, 8],
    'n_iter': [5, 10, 20],
    'algorithm': ['arpack', 'randomized'],
    'tol': [1e-8, 1e-9, 1e-10],
}


def write_ids(lines, fn):
    with open(fn, 'w') as fh:
        for line in lines:
            fh.write("%s\n" % line)


def read_ids(fn):
    result = []
    with open(fn, 'r') as fh:
        for line in fh:
            result.append(int(line))
    return result


def read_artist_alias(fn):

    uf = UnionFind()
    with open(fn, "r") as fh:
        for line in fh:
            try:
                a, b = map(int, line.split("\t"))
            except ValueError:
                continue
            uf.union(a, b)

    result = {}
    for s in uf.sets():
        sorted_set = sorted(s)
        head = sorted_set[0]
        for remaining in sorted_set[1:]:
            result[remaining] = head

    return result


def read_artist_data(fn, alias):

    result = {}
    with open(fn, "r") as fh:
        for line in fh:
            a, b = line.split("\t")
            a = int(a)
            result[alias.get(a, a)] = b
    return result


def read_user_artist(fn, alias):

    result = defaultdict(partial(defaultdict, float))
    with open(fn, "r") as fh:
        for line in fh:
            a, b, c = line.split(" ")
            a, b, c = int(a), int(b), float(c)
            result[a][alias.get(b, b)] += c

    ratings = []
    for uid, vals in result.iteritems():
        for pid, rating in vals.iteritems():
            ratings.append(Rating(uid, pid, rating))

    return np.asarray(ratings, dtype=Rating)


def ratings2coo(ratings):
    result = defaultdict(partial(defaultdict, float))
    for uid, pid, rating in ratings:
        result[uid][pid] += rating
    return dd2coo(result)


def buildParamGrid(gridSpec):
    ks, vs = zip(*gridSpec.items())
    all_params = []

    # ensure all vs are iterable
    all_vals = []
    for v in vs:
        if not isiterable(v):
            v = [v]
        all_vals.append(v)

    for vv in product(*all_vals):
        these_params = []
        for k, v in zip(ks, vv):
            these_params.append((k, v))
        all_params.append(these_params)
    return all_params


def group_ratings(ratings, by='row'):
    """Given array of ratings, group them by left column
    """
    dd = defaultdict(partial(defaultdict, float))
    idx0, idx1 = (0, 1) if by == 'row' else (1, 0)
    for rating in ratings:
        dd[rating[idx0]][rating[idx1]] += rating[2]
    return dd


class Model(object):

    def __init__(self, params):
        self.row_map = None
        self.col_map = None
        self.params = params
        self.model_ = None
        self.components_ = None

    def predict(self, ratings, order='inner'):
        """Make predictions given a list of ratings/tuples

        If order='inner' (default), will only return rows *and* columns present
        in the testing input. If order='row', will limit rows to those present
        in the input, and return all columns. If order='column', vice versa.
        """
        row_vecs, col_vecs = self.components_
        row_map, col_map = self.row_map, self.col_map

        red_rows = row_map.keys() \
            if order == 'column' \
            else [r for r in set(ratings[:, 0]) if r in row_map]
        red_cols = col_map.keys() \
            if order == 'row' \
            else [c for c in set(ratings[:, 1]) if c in col_map]
        row_results = sorted([(row_map[r], r) for r in red_rows])
        col_results = sorted([(col_map[c], c) for c in red_cols])
        row_idx, row_names = map(list, zip(*row_results) if row_results else ((), ()))
        col_idx, col_names = map(list, zip(*col_results) if col_results else ((), ()))

        act_row_vecs = row_vecs[row_idx]
        act_col_vecs = col_vecs[col_idx]
        result = np.matmul(act_row_vecs, act_col_vecs.T)
        row_map = {k: idx for idx, k in enumerate(row_names)}
        col_map = {k: idx for idx, k in enumerate(col_names)}
        return row_map, col_map, result

    def compute_roc(self, ratings, normalize=False):
        """Compute Receiver Operating Characteristic Curve
        """
        row_map, col_map, preds = self.predict(ratings, order='row')
        inv_col_map = {v: k for k, v in col_map.iteritems()}

        ordered_ratings = []
        # rebuild ratings array s.t. everything is ordered
        grouped_ratings = group_ratings(ratings, by='row')
        # limit to only those rows for which ground truth values are available
        for row_name, row_ratings in grouped_ratings.iteritems():
            if row_name not in row_map:
                continue
            row_idx = row_map[row_name]
            predicted = preds[row_idx]
            new_row = []
            # for every predicted rating, get corresponding ground truth
            # value (set to zero if not found)
            for col_idx, val in enumerate(predicted):
                col_name = inv_col_map.get(col_idx)
                rating = row_ratings.get(col_name, 0.0)
                new_row.append((val, rating))
            # sort ground truth ratings by their predicted
            # values (order will become important)
            vec = np.array(zip(*sorted(new_row, reverse=True))[1])
            ordered_ratings.append(vec)
        # convert to NumPy array
        pos = np.asarray(ordered_ratings)
        # sum all rows s.t. resulting vector of row totals is a column
        sum_col = np.expand_dims(pos.sum(axis=1), 1)
        if normalize:
            # normalize all values by row totals and replace any NaNs with zeros
            pos = np.nan_to_num(pos / sum_col)
            # get negative counts
            neg = 1.0 - pos
        else:
            neg = sum_col - pos
        # get column totals for positives and negatives
        vsum_pos = pos.sum(axis=0)
        vsum_neg = neg.sum(axis=0)
        # normalize column totals of positives and negatives by their
        # respective grand totals
        vsum_pos /= vsum_pos.sum()
        vsum_neg /= vsum_neg.sum()
        # return cumulative sums from left to right for positives
        # and negatives
        xs = np.cumsum(vsum_neg)
        ys = np.cumsum(vsum_pos)
        # the last value will be correct (equal to 1.0) due to normalization
        # followed by cumulative sum, however we need to prepend the first
        # value (zero)
        fpr = np.insert(xs, 0, 0.0)
        tpr = np.insert(ys, 0, 0.0)
        return fpr, tpr

    def loss_rmse(self, ratings):
        """Calculate model loss on validation set
        """
        user_map, prod_map, preds = self.predict(ratings)
        total = 0.0
        n = 0
        for u, p, r in ratings:
            if u not in user_map:
                continue
            if p not in prod_map:
                continue
            r_pred = preds[user_map[u], prod_map[p]]
            total += (r - r_pred) ** 2
            n += 1
        return np.sqrt(total / n)

    def loss(self, ratings, loss='rmse'):
        """Calculate model loss on validation set
        """
        if loss == 'rmse':
            return self.loss_rmse(ratings)


class PMFModel(Model):

    def fit(self, data):
        self.row_map, self.col_map, coo_mat = ratings2coo(data)
        param_string = ' '.join(["-%s %s" % pair for pair in self.params])
        self.model_ = model = libpmf.train(coo_mat, param_string)
        self.components_ = model['W'], model['H']
        return self


class NMFModel(Model):

    def fit(self, data):
        param_dict = dict(self.params)
        nmf = decomposition.NMF(**param_dict)
        row_map, col_map, coo_mat = ratings2coo(data)
        self.row_map, self.col_map = row_map, col_map
        self.model_ = nmf.fit(coo_mat)
        comp2 = nmf.components_.T
        self.components_ = coo_mat.dot(comp2), comp2
        return self


class SVDModel(Model):

    def fit(self, data):
        param_dict = dict(self.params)
        nmf = decomposition.TruncatedSVD(**param_dict)
        row_map, col_map, coo_mat = ratings2coo(data)
        self.row_map, self.col_map = row_map, col_map
        self.model_ = nmf.fit(coo_mat)
        comp2 = nmf.components_.T
        self.components_ = coo_mat.dot(comp2), comp2
        return self


def gridSearch(model_class, trainval, params, n_folds=5, seed=42):
    folds = KFold(len(trainval), n_folds=n_folds, shuffle=True, random_state=seed)
    results = []
    total = len(params)
    for idx, param in enumerate(params):
        fold_losses = []
        training_times = []
        for training_idx, validation_idx in folds:
            with PMTimer() as timer:
                model = model_class(param).fit(trainval[training_idx])
            training_times.append(timer.clock_interval)
            fold_losses.append(model.loss(trainval[validation_idx]))
        avg_loss = np.average(fold_losses)
        avg_time = np.average(training_times)
        logger.info("%d/%d: loss=%.3f, param=%s (%.3f sec)", idx, total, avg_loss, param, avg_time)
        results.append((avg_loss, avg_time, param))
    # refit the model on the best parameter set
    best_loss, best_time, best_param = min(results)
    logger.info("best: loss=%.3f, time=%.3f sec, param=%s", best_loss, best_time, best_param)
    return model_class(best_param).fit(trainval)


MODEL_CLASSES = {
    'PMF-NMF': (PMFModel, pmf_nmf_grid),
    'PMF-SVD': (PMFModel, pmf_svd_grid),
    'SKL-NMF': (NMFModel, skl_nmf_grid),
    'SKL-SVD': (SVDModel, skl_svd_grid),
}


def parse_args(args=None):
    parser = argparse.ArgumentParser(args)
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Data directory')
    parser.add_argument('--setting', type=str, choices=MODEL_CLASSES.keys(),
                        default='SKL-SVD', help='which setting to use')
    parser.add_argument('--seed', type=int, default=42,
                        help='random state')
    parser.add_argument('--normalize_roc', type=int, default=1,
                        help='set to 1 to normalize ROC, 0 otherwise')
    parser.add_argument('--folds', type=int, default=5,
                        help='number of folds in cross-validation')
    namespace = parser.parse_args()
    return namespace


def build_model(args, trainval):
    model_class, model_grid = MODEL_CLASSES[args.setting]
    paramGrid = buildParamGrid(model_grid)
    return gridSearch(model_class, trainval, paramGrid, n_folds=args.folds, seed=args.seed)


def plot_roc(xs, ys, save_to=None):

    import matplotlib.pyplot as plt

    auc_score = metrics.auc(xs, ys, reorder=False)
    fig, ax = plt.subplots(1)

    ax.plot(xs, ys, '-', color='blue')
    ax.plot([0.0, 1.0], [0.0, 1.0], '--', color='gray')
    ax.set_ylabel("True Positive Rate")
    ax.set_xlabel("False Positive Rate")
    ax.set_title("ROC Curve (AUC=%.3f)" % auc_score)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])

    if save_to is None:
        fig.show()
    else:
        fig.savefig(save_to)
    fig.clf()


def run(args):
    alias = read_artist_alias(os.path.join(args.data_dir, "artist_alias.txt"))
    ratings_array = read_user_artist(os.path.join(args.data_dir, "user_artist_data.txt"), alias)

    trainval, testing = train_test_split(ratings_array, test_size=0.2, random_state=args.seed)

    model = build_model(args, trainval)

    # final evaluation on the testing set
    logger.info("testing loss: %.3f", model.loss(testing))

    xs, ys = model.compute_roc(testing, normalize=args.normalize_roc)
    plot_roc(xs, ys, save_to="fig-roc-norm-%s.png" % args.setting)


if __name__ == "__main__":
    run(parse_args())
