

import torch
import numpy as np
from math import pi
from scipy.special import logsumexp
from sklearn.covariance import LedoitWolf
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
from sklearn.cluster import KMeans
import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
import seaborn as sns
sns.set(style="white", font="Arial")
colors = sns.color_palette("Paired", n_colors=12).as_hex()


def calculate_matmul_n_times(n_components, mat_a, mat_b):
    """
    Calculate matrix product of two matrics with mat_a[0] >= mat_b[0].
    Bypasses torch.matmul to reduce memory footprint.
    args:
        mat_a:      torch.Tensor (n, k, 1, d)
        mat_b:      torch.Tensor (1, k, d, d)
    """
    res = torch.zeros(mat_a.shape).to(mat_a.device)

    for i in range(n_components):
        mat_a_i = mat_a[:, i, :, :].squeeze(-2)
        mat_b_i = mat_b[0, i, :, :].squeeze()
        res[:, i, :, :] = mat_a_i.mm(mat_b_i).unsqueeze(1)

    return res


def calculate_matmul(mat_a, mat_b):
    """
    Calculate matrix product of two matrics with mat_a[0] >= mat_b[0].
    Bypasses torch.matmul to reduce memory footprint.
    args:
        mat_a:      torch.Tensor (n, k, 1, d)
        mat_b:      torch.Tensor (n, k, d, 1)
    """
    assert mat_a.shape[-2] == 1 and mat_b.shape[-1] == 1
    return torch.sum(mat_a.squeeze(-2) * mat_b.squeeze(-1), dim=2, keepdim=True)


def euclidean_metric_np(X, centroids):
    X = np.expand_dims(X, 1)
    centroids = np.expand_dims(centroids, 0)
    dists = (X - centroids) ** 2
    dists = np.sum(dists, axis=2)
    return dists


def euclidean_metric_gpu(X, centers):
    X = X.unsqueeze(1)
    centers = centers.unsqueeze(0)

    dist = torch.sum((X - centers) ** 2, dim=-1)
    return dist


def kmeans_fun_gpu(X, K=10, max_iter=1000, batch_size=8096, tol=1e-40):
    N = X.shape[0]

    indices = torch.randperm(N)[:K]
    init_centers = X[indices]

    batchs = N // batch_size
    last = 1 if N % batch_size != 0 else 0

    choice_cluster = torch.zeros([N]).cuda()  #TODO: turn to layer to avoid initialization maybe? maybe not possible because N changes...understand better
    for _ in range(max_iter):
        for bn in range(batchs + last):
            if bn == batchs and last == 1:
                _end = -1
            else:
                _end = (bn + 1) * batch_size
            X_batch = X[bn * batch_size: _end]

            dis_batch = euclidean_metric_gpu(X_batch, init_centers)
            choice_cluster[bn * batch_size: _end] = torch.argmin(dis_batch, dim=1)

        init_centers_pre = init_centers.clone()
        for index in range(K):
            selected = torch.nonzero(choice_cluster == index).squeeze().cuda()
            selected = torch.index_select(X, 0, selected)
            init_centers[index] = selected.mean(dim=0)

        center_shift = torch.sum(
            torch.sqrt(
                torch.sum((init_centers - init_centers_pre) ** 2, dim=1)
            ))
        if center_shift < tol:
            break

    #TODO: there's no need to turn to numpy!
    k_mean = init_centers.detach().cpu().numpy()
    choice_cluster = choice_cluster.detach().cpu().numpy()
    return k_mean, choice_cluster


def _cal_var(X, centers=None, choice_cluster=None, K=10):
    D = X.shape[1]
    k_var = np.zeros([K, D, D])
    eps = np.eye(D) * 1e-10
    if centers is not None:
        _dist = euclidean_metric_np(X, centers)
        choice_cluster = np.argmin(_dist, axis=1)

    for k in range(K):
        samples = X[k == choice_cluster]
        _m = np.mean(samples, axis=0)
        k_var[k] = LedoitWolf().fit(samples).covariance_ + eps  #TODO: is there a way to do it in torch? well this is stupid!!!! turning to numpy every iteration!!!!
    return k_var.astype(np.float32)


def mahalanobias_metric_gpu(X, mean, var):
    torch.cuda.empty_cache()
    dis = torch.zeros([X.shape[0], mean.shape[0]])
    for k in range(mean.shape[0]):
        _m = mean[k]
        _inv = torch.inverse(var[k])
        # method 1
        delta = X - _m
        temp = torch.mm(delta, _inv)
        dis[:, k] = torch.sqrt_(torch.sum(torch.mul(delta, temp), dim=1))
    return dis


def check_nan(x):
    isnan = torch.isnan(x).int()
    loc = isnan.sum()
    # print(f"any nan: {loc.item()}")



class GMM(object):
    def __init__(self, K=5, type='full'):
        '''
        Initlize GMM
        :param K: number of clusters
        :param type:
        '''
        self.K = K
        self.type = type

        self.eps = 1e-10
        self.log2pi = np.log(2 * np.pi)

    def _logpdf(self):
        '''
        X： N x D
        mu: K x D
        var: K x D x D
        alpha: 1 x K
        :return: log_prob
        '''
        log_prob = torch.zeros([self.N, self.K]).cuda()
        for k in range(self.K):
            mu_k = self.mu[k].unsqueeze(0)
            var_k = self.var[k]
            var_k_inv = torch.inverse(var_k)

            det_var = torch.det(var_k)

            delta = self.X - mu_k
            temp = torch.mm(delta, var_k_inv)
            dist = torch.sum(torch.mul(delta, temp), dim=1)

            log_prob_k = -0.5 * (self.D * self.log2pi + torch.log(det_var) + dist) + torch.log(self.alpha[k])
            log_prob[:, k] = log_prob_k

        return log_prob

    def _pdf(self):
        '''
        X： N x D
        mu: K x D
        var: K x D x D
        alpha: 1 x K
        :return:
        '''

        self.log_prob = self._logpdf()
        max_log_prob = -torch.max(self.log_prob, dim=1, keepdim=True)[0]
        log_prob = self.log_prob / max_log_prob
        self.prob = torch.exp(log_prob)

        check_nan(self.prob)
        print(self.alpha)
        print(f"{torch.max(self.prob)}, {torch.min(self.prob)}")

        return self.prob

    def _e_step(self):
        '''
        prob: N x K
        '''
        self.prob = self._pdf()
        prob_sum = torch.sum(self.prob, dim=1, keepdim=True)
        self.prob = self.prob / prob_sum

        check_nan(self.prob)

        return self.prob

    def _m_step(self):
        '''
        X： N x D
        mu: K x D
        var: K x D x D
        alpha: 1 x K
        prob: N x K
        '''

        self.alpha = torch.sum(self.prob, dim=0)
        for k in range(self.K):
            prob_k = self.prob[:, k].unsqueeze(1)
            self.mu[k] = torch.sum(self.X * prob_k, dim=0) / self.alpha[k]

            mu_k = self.mu[k].unsqueeze(0)
            delta = self.X - mu_k  # N x D
            delta_t = torch.transpose(delta, 0, 1)  # D x N
            delta = delta * prob_k

            self.var[k] = torch.mm(delta_t, delta) / self.alpha[k] + self.eps_mat

        self.alpha = self.alpha / self.N

    def fit(self, X, max_iters=200, tol=1e-60):
        '''
        fit the X to the GMM model
        :param X: N x D
        :param max_iters:
        :return:
        '''
        self.X = X
        self.N, self.D = X.shape[0], X.shape[1]
        self.pi = np.power(np.pi * 2, self.D / 2)
        self.eps_mat = torch.eye(self.D).cuda() * self.eps

        # Initilize parameters by k-means
        init_centers, _ = kmeans_fun_gpu(X, K=self.K)
        self.mu = torch.from_numpy(init_centers.astype(np.float32)).cuda()

        self.var = _cal_var(X.detach().cpu().numpy(), centers=init_centers, K=K)
        self.var = torch.from_numpy(self.var).cuda()

        # self.mu = torch.randn(self.K, self.D).cuda()
        # var = torch.eye(self.D)
        # self.var = var.expand(self.K, self.D, self.D).cuda()

        self.alpha = torch.ones([self.K, 1]) / self.K
        self.alpha = self.alpha.cuda()

        log_lh_old = 0
        for iter in range(max_iters):
            # print(f"GMM Step {iter + 1} ...")
            self._e_step()
            self._m_step()
            log_lh = -torch.sum(self.log_prob)
            if iter >= 1 and torch.abs(log_lh - log_lh_old) < tol:
                break
            log_lh_old = log_lh
            print(f"[!!!] Iter-{iter + 1} log likely hood: {log_lh.item():.8f}")

        prob = self._e_step()
        pred = torch.argmax(prob, dim=1)
        return self.mu, pred


class GMM_Batch(object):
    def __init__(self, K=5, type='full'):
        '''
        Initlize GMM
        :param K: number of clusters
        :param type:
        '''
        self.K = K
        self.type = type

        self.eps = 1e-10
        self.log2pi = np.log(2 * np.pi)

        self.reset_batch_cache()

    def reset_batch_cache(self):
        self.cache = {"prob": [],
                      "mu": [[] for _ in range(self.K)],
                      "var": [[] for _ in range(self.K)],
                      "alpha": []}

    def _logpdf(self, x):
        '''
        X： N x D
        mu: K x D
        var: K x D x D
        alpha: 1 x K
        :return: log_prob
        '''
        log_prob = torch.zeros([x.shape[0], self.K]).cuda()
        for k in range(self.K):
            mu_k = self.mu[k].unsqueeze(0)
            var_k = self.var[k]
            var_k_inv = torch.inverse(var_k)

            det_var = torch.det(var_k)

            delta = x - mu_k
            temp = torch.mm(delta, var_k_inv)
            dist = torch.sum(torch.mul(delta, temp), dim=1)

            log_prob_k = -0.5 * (self.D * self.log2pi + torch.log(det_var) + dist) + torch.log(self.alpha[k])
            log_prob[:, k] = log_prob_k

        return log_prob

    def _pdf(self, x):
        '''
        X： N x D
        mu: K x D
        var: K x D x D
        alpha: 1 x K
        :return:
        '''

        self.log_prob = self._logpdf(x)
        max_log_prob = -torch.max(self.log_prob, dim=1, keepdim=True)[0]
        log_prob = self.log_prob / max_log_prob
        self.prob = torch.exp(log_prob)

        # check_nan(self.prob)
        # print(self.alpha)
        # print(f"{torch.max(self.prob)}, {torch.min(self.prob)}")
        return self.prob

    def _e_step(self, x):
        '''
        prob: N x K
        '''
        self.prob = self._pdf(x)
        prob_sum = torch.sum(self.prob, dim=1, keepdim=True) + self.eps
        self.prob = self.prob / prob_sum

        check_nan(self.prob)

        self.cache['prob'].append(self.prob)
        return self.prob

    def _m_step(self, x):
        '''
        X： N x D
        mu: K x D
        var: K x D x D
        alpha: 1 x K
        prob: N x K
        '''

        self.alpha = torch.sum(self.prob, dim=0)
        for k in range(self.K):
            prob_k = self.prob[:, k].unsqueeze(1)
            # self.mu[k] = torch.sum(x * prob_k, dim=0) / self.alpha[k]
            self.cache['mu'][k].append(torch.sum(x * prob_k, dim=0).unsqueeze(0))

            mu_k = self.mu[k].unsqueeze(0)
            delta = x - mu_k  # N x D
            delta_t = torch.transpose(delta, 0, 1)  # D x N
            delta = delta * prob_k

            # self.var[k] = torch.mm(delta_t, delta) / self.alpha[k] + self.eps_mat
            self.cache['var'][k].append(torch.mm(delta_t, delta).unsqueeze(0))

        # self.alpha = self.alpha / self.N
        self.cache['alpha'].append(self.alpha.unsqueeze(0))

    def fit(self, X, batch_size=1024, max_iters=200, tol=1e-60):
        '''
        fit the X to the GMM model
        :param X: N x D
        :param max_iters:
        :return:
        '''
        self.N, self.D = X.shape[0], X.shape[1]
        self.pi = np.power(np.pi * 2, self.D / 2)
        self.eps_mat = torch.eye(self.D).cuda() * self.eps  #TODO: turn to layer to avoid initialization

        # Initilize parameters by k-means
        init_centers, _ = kmeans_fun_gpu(X, K=self.K)
        self.mu = torch.from_numpy(init_centers.astype(np.float32)).cuda()

        self.var = _cal_var(X.detach().cpu().numpy(), centers=init_centers, K=self.K)  #TODO: wait what?! why?!
        self.var = torch.from_numpy(self.var).cuda()


        self.alpha = torch.ones([self.K, 1]) / self.K
        self.alpha = self.alpha.cuda()

        batchs = self.N // batch_size
        last = 1 if self.N % batch_size != 0 else 0
        log_lh_old = 0
        for iter in range(max_iters):
            for bn in range(batchs + last):
                if bn == batchs and last == 1:
                    _end = -1
                else:
                    _end = (bn + 1) * batch_size
                X_batch = X[bn * batch_size: _end]

                self._e_step(X_batch)
                self._m_step(X_batch)

            ##############+ update mean and covariance +################
            self.alpha = torch.cat(self.cache['alpha'], dim=0)
            self.alpha = torch.sum(self.alpha, dim=0)

            for k in range(self.K):
                mu_k = torch.cat(self.cache['mu'][k], dim=0)
                self.mu[k] = torch.sum(mu_k, dim=0) / self.alpha[k]
                var_k = torch.cat(self.cache['var'][k], dim=0)
                self.var[k] = torch.sum(var_k, dim=0) / self.alpha[k] + self.eps_mat
            self.alpha = self.alpha / self.N
            ##############++++++++++++++++++++++++++++++################

            prob = torch.cat(self.cache['prob'], dim=0)
            log_lh = -torch.sum(prob.log())
            self.reset_batch_cache()
            if iter >= 1 and torch.abs(log_lh - log_lh_old) < tol:
                break
            log_lh_old = log_lh
            print(f"[!!!] Iter-{iter + 1} log likely hood: {log_lh.item():.8f}")

        # prob = self._e_step(X)
        pred = torch.argmax(prob, dim=1)
        return self.mu, pred


def bla_GMM_testing():

    n = 1000
    n1 = 500
    K = 5
    data, label = make_blobs(n_samples=n, n_features=512, centers=K)
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(1, 3, 1, projection='3d', facecolor='white')
    ax.scatter(data[:n1, 0], data[:n1, 1], data[:n1, 2], c=label[:n1])
    ax.set_title("Data")

    X = torch.from_numpy(data.astype(np.float32)).cuda()

    st = time.time()
    mean, pre_label = kmeans_fun_gpu(X, K, max_iter=1000)
    et = time.time()
    print(f"KMeans-Batch-pytorch fitting time: {(et - st):.3f}ms")
    ax = fig.add_subplot(1, 3, 2, projection='3d', facecolor='white')
    ax.scatter(data[:n1, 0], data[:n1, 1], data[:n1, 2], c=pre_label[:n1])
    ax.set_title(f"KM-B:{(et - st):.1f}ms")

    st = time.time()
    gmm = GMM_Batch(K=K)
    _, pre_label = gmm.fit(X, batch_size=100, max_iters=100)
    pre_label = pre_label.detach().cpu().numpy()
    print(gmm.alpha)
    et = time.time()
    print(f"GMM-Batch-pytorch fitting time: {(et - st):.3f}ms")
    ax = fig.add_subplot(1, 3, 3, projection='3d', facecolor='white')
    ax.scatter(data[:n1, 0], data[:n1, 10], data[:n1, 20], c=pre_label[:n1])
    ax.set_title(f"GMM-B:{(et - st):.1f}ms")

    plt.show()


def bla_KMeans_testing():
    n = 500000
    n1 = 5000
    K = 5
    data, label = make_blobs(n_samples=n, n_features=3, centers=K)
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(1, 3, 1, projection='3d', facecolor='white')
    ax.scatter(data[:n1, 0], data[:n1, 1], data[:n1, 2], c=label[:n1])
    ax.set_title("Data")


    model = KMeans(n_clusters=K, max_iter=100, tol=1e-40)
    st = time.time()
    model.fit_transform(data, label)
    et = time.time()
    print(f"Sklearn KMeans fitting time: {(et-st):.3f}ms")
    sk_pred_label = model.predict(data)
    ax = fig.add_subplot(1, 3, 2, projection='3d', facecolor='white')
    ax.scatter(data[:n1, 0], data[:n1, 1], data[:n1, 2], c=sk_pred_label[:n1])
    ax.set_title(f"S-KM:{(et-st):.1f}ms")

    X = torch.from_numpy(data.astype(np.float32)).cuda()
    st = time.time()
    mean, pre_label = kmeans_fun_gpu(X, K=K, max_iter=30, batch_size=8096, tol=1e-40)
    et = time.time()
    print(f"KMeans-Batch-pytorch fitting time: {(et-st):.3f}ms")
    ax = fig.add_subplot(1, 3, 3, projection='3d', facecolor='white')
    ax.scatter(data[:n1, 0], data[:n1, 1], data[:n1, 2], c=pre_label[:n1])
    ax.set_title(f"KM-B:{(et-st):.1f}ms")

    plt.show()


# bla_KMeans_testing()
# bla_GMM_testing()

##############################################################################################################################################



##############################################################################################################################################
### Another GMM Implementation: ###
class GaussianMixture(torch.nn.Module):
    """
    Fits a mixture of k=1,..,K Gaussians to the input data (K is supplied via n_components).
    Input tensors are expected to be flat with dimensions (n: number of samples, d: number of features).
    The model then extends them to (n, 1, d).
    The model parametrization (mu, sigma) is stored as (1, k, d),
    probabilities are shaped (n, k, 1) if they relate to an individual sample,
    or (1, k, 1) if they assign membership probabilities to one of the mixture components.
    """

    def __init__(self, n_components, n_features, covariance_type="full", eps=1.e-6, init_params="kmeans", mu_init=None, var_init=None):
        """
        Initializes the model and brings all tensors into their required shape.
        The class expects data to be fed as a flat tensor in (n, d).
        The class owns:
            x:               torch.Tensor (n, 1, d)
            mu:              torch.Tensor (1, k, d)
            var:             torch.Tensor (1, k, d) or (1, k, d, d)
            pi:              torch.Tensor (1, k, 1)
            covariance_type: str
            eps:             float
            init_params:     str
            log_likelihood:  float
            n_components:    int
            n_features:      int
        args:
            n_components:    int
            n_features:      int
        options:
            mu_init:         torch.Tensor (1, k, d)
            var_init:        torch.Tensor (1, k, d) or (1, k, d, d)
            covariance_type: str
            eps:             float
            init_params:     str
        """
        super(GaussianMixture, self).__init__()

        self.n_components = n_components
        self.n_features = n_features

        self.mu_init = mu_init
        self.var_init = var_init
        self.eps = eps

        self.log_likelihood = -np.inf

        self.covariance_type = covariance_type
        self.init_params = init_params

        assert self.covariance_type in ["full", "diag"]
        assert self.init_params in ["kmeans", "random"]

        self._init_params()

    def _init_params(self):
        if self.mu_init is not None:
            assert self.mu_init.size() == (1, self.n_components, self.n_features), "Input mu_init does not have required tensor dimensions (1, %i, %i)" % (self.n_components, self.n_features)
            # (1, k, d)
            self.mu = torch.nn.Parameter(self.mu_init, requires_grad=False)
        else:
            self.mu = torch.nn.Parameter(torch.randn(1, self.n_components, self.n_features), requires_grad=False)

        if self.covariance_type == "diag":
            if self.var_init is not None:
                # (1, k, d)
                assert self.var_init.size() == (1, self.n_components, self.n_features), "Input var_init does not have required tensor dimensions (1, %i, %i)" % (self.n_components, self.n_features)
                self.var = torch.nn.Parameter(self.var_init, requires_grad=False)
            else:
                self.var = torch.nn.Parameter(torch.ones(1, self.n_components, self.n_features), requires_grad=False)
        elif self.covariance_type == "full":
            if self.var_init is not None:
                # (1, k, d, d)
                assert self.var_init.size() == (1, self.n_components, self.n_features, self.n_features), "Input var_init does not have required tensor dimensions (1, %i, %i, %i)" % (self.n_components, self.n_features, self.n_features)
                self.var = torch.nn.Parameter(self.var_init, requires_grad=False)
            else:
                self.var = torch.nn.Parameter(
                    torch.eye(self.n_features).reshape(1, 1, self.n_features, self.n_features).repeat(1, self.n_components, 1, 1),
                    requires_grad=False
                )

        # (1, k, 1)
        self.pi = torch.nn.Parameter(torch.Tensor(1, self.n_components, 1), requires_grad=False).fill_(1. / self.n_components)
        self.params_fitted = False

    def check_size(self, x):
        if len(x.size()) == 2:
            # (n, d) --> (n, 1, d)
            x = x.unsqueeze(1)

        return x

    def bic(self, x):
        """
        Bayesian information criterion for a batch of samples.
        args:
            x:      torch.Tensor (n, d) or (n, 1, d)
        returns:
            bic:    float
        """
        x = self.check_size(x)
        n = x.shape[0]

        # Free parameters for covariance, means and mixture components
        free_params = self.n_features * self.n_components + self.n_features + self.n_components - 1

        bic = -2. * self.__score(x, as_average=False).mean() * n + free_params * np.log(n)

        return bic

    def fit(self, x, delta=1e-3, n_iter=100, warm_start=False):
        """
        Fits model to the data.
        args:
            x:          torch.Tensor (n, d) or (n, k, d)
        options:
            delta:      float
            n_iter:     int
            warm_start: bool
        """
        if not warm_start and self.params_fitted:
            self._init_params()

        x = self.check_size(x)

        if self.init_params == "kmeans" and self.mu_init is None:
            mu = self.get_kmeans_mu(x, n_centers=self.n_components)
            self.mu.data = mu

        i = 0
        j = np.inf

        while (i <= n_iter) and (j >= delta):

            log_likelihood_old = self.log_likelihood
            mu_old = self.mu
            var_old = self.var

            self.__em(x)
            self.log_likelihood = self.__score(x)

            if torch.isinf(self.log_likelihood.abs()) or torch.isnan(self.log_likelihood):
                device = self.mu.device
                # When the log-likelihood assumes unbound values, reinitialize model
                self.__init__(self.n_components,
                              self.n_features,
                              covariance_type=self.covariance_type,
                              mu_init=self.mu_init,
                              var_init=self.var_init,
                              eps=self.eps)
                for p in self.parameters():
                    p.data = p.data.to(device)
                if self.init_params == "kmeans":
                    self.mu.data, = self.get_kmeans_mu(x, n_centers=self.n_components)

            i += 1
            j = self.log_likelihood - log_likelihood_old

            if j <= delta:
                # When score decreases, revert to old parameters
                self.__update_mu(mu_old)
                self.__update_var(var_old)

        self.params_fitted = True

    def predict(self, x, probs=False):
        """
        Assigns input data to one of the mixture components by evaluating the likelihood under each.
        If probs=True returns normalized probabilities of class membership.
        args:
            x:          torch.Tensor (n, d) or (n, 1, d)
            probs:      bool
        returns:
            p_k:        torch.Tensor (n, k)
            (or)
            y:          torch.LongTensor (n)
        """
        x = self.check_size(x)

        weighted_log_prob = self._estimate_log_prob(x) + torch.log(self.pi)

        if probs:
            p_k = torch.exp(weighted_log_prob)
            return torch.squeeze(p_k / (p_k.sum(1, keepdim=True)))
        else:
            return torch.squeeze(torch.max(weighted_log_prob, 1)[1].type(torch.LongTensor))

    def predict_proba(self, x):
        """
        Returns normalized probabilities of class membership.
        args:
            x:          torch.Tensor (n, d) or (n, 1, d)
        returns:
            y:          torch.LongTensor (n)
        """
        return self.predict(x, probs=True)

    def sample(self, n):
        """
        Samples from the model.
        args:
            n:          int
        returns:
            x:          torch.Tensor (n, d)
            y:          torch.Tensor (n)
        """
        counts = torch.distributions.multinomial.Multinomial(total_count=n, probs=self.pi.squeeze()).sample()
        x = torch.empty(0, device=counts.device)
        y = torch.cat([torch.full([int(sample)], j, device=counts.device) for j, sample in enumerate(counts)])

        # Only iterate over components with non-zero counts
        for k in np.arange(self.n_components)[counts > 0]:
            if self.covariance_type == "diag":
                x_k = self.mu[0, k] + torch.randn(int(counts[k]), self.n_features, device=x.device) * torch.sqrt(self.var[0, k])
            elif self.covariance_type == "full":
                d_k = torch.distributions.multivariate_normal.MultivariateNormal(self.mu[0, k], self.var[0, k])
                x_k = torch.stack([d_k.sample() for _ in range(int(counts[k]))])

            x = torch.cat((x, x_k), dim=0)

        return x, y

    def score_samples(self, x):
        """
        Computes log-likelihood of samples under the current model.
        args:
            x:          torch.Tensor (n, d) or (n, 1, d)
        returns:
            score:      torch.LongTensor (n)
        """
        x = self.check_size(x)

        score = self.__score(x, as_average=False)
        return score

    def _estimate_log_prob(self, x):
        """
        Returns a tensor with dimensions (n, k, 1), which indicates the log-likelihood that samples belong to the k-th Gaussian.
        args:
            x:            torch.Tensor (n, d) or (n, 1, d)
        returns:
            log_prob:     torch.Tensor (n, k, 1)
        """
        x = self.check_size(x)

        if self.covariance_type == "full":
            mu = self.mu
            var = self.var

            precision = torch.inverse(var)
            d = x.shape[-1]

            log_2pi = d * np.log(2. * pi)

            log_det = self._calculate_log_det(precision)

            x_mu_T = (x - mu).unsqueeze(-2)
            x_mu = (x - mu).unsqueeze(-1)

            x_mu_T_precision = calculate_matmul_n_times(self.n_components, x_mu_T, precision)
            x_mu_T_precision_x_mu = calculate_matmul(x_mu_T_precision, x_mu)

            return -.5 * (log_2pi - log_det + x_mu_T_precision_x_mu)

        elif self.covariance_type == "diag":
            mu = self.mu
            prec = torch.rsqrt(self.var)

            log_p = torch.sum((mu * mu + x * x - 2 * x * mu) * (prec ** 2), dim=2, keepdim=True)
            log_det = torch.sum(torch.log(prec), dim=2, keepdim=True)

            return -.5 * (self.n_features * np.log(2. * pi) + log_p) + log_det

    def _calculate_log_det(self, var):
        """
        Calculate log determinant in log space, to prevent overflow errors.
        args:
            var:            torch.Tensor (1, k, d, d)
        """
        log_det = torch.empty(size=(self.n_components,)).to(var.device)

        for k in range(self.n_components):
            log_det[k] = 2 * torch.log(torch.diagonal(torch.linalg.cholesky(var[0, k]))).sum()

        return log_det.unsqueeze(-1)

    def _e_step(self, x):
        """
        Computes log-responses that indicate the (logarithmic) posterior belief (sometimes called responsibilities) that a data point was generated by one of the k mixture components.
        Also returns the mean of the mean of the logarithms of the probabilities (as is done in sklearn).
        This is the so-called expectation step of the EM-algorithm.
        args:
            x:              torch.Tensor (n, d) or (n, 1, d)
        returns:
            log_prob_norm:  torch.Tensor (1)
            log_resp:       torch.Tensor (n, k, 1)
        """
        x = self.check_size(x)

        weighted_log_prob = self._estimate_log_prob(x) + torch.log(self.pi)

        log_prob_norm = torch.logsumexp(weighted_log_prob, dim=1, keepdim=True)
        log_resp = weighted_log_prob - log_prob_norm

        return torch.mean(log_prob_norm), log_resp

    def _m_step(self, x, log_resp):
        """
        From the log-probabilities, computes new parameters pi, mu, var (that maximize the log-likelihood). This is the maximization step of the EM-algorithm.
        args:
            x:          torch.Tensor (n, d) or (n, 1, d)
            log_resp:   torch.Tensor (n, k, 1)
        returns:
            pi:         torch.Tensor (1, k, 1)
            mu:         torch.Tensor (1, k, d)
            var:        torch.Tensor (1, k, d)
        """
        x = self.check_size(x)

        resp = torch.exp(log_resp)

        pi = torch.sum(resp, dim=0, keepdim=True) + self.eps
        mu = torch.sum(resp * x, dim=0, keepdim=True) / pi

        if self.covariance_type == "full":
            eps = (torch.eye(self.n_features) * self.eps).to(x.device)
            var = torch.sum((x - mu).unsqueeze(-1).matmul((x - mu).unsqueeze(-2)) * resp.unsqueeze(-1), dim=0,
                            keepdim=True) / torch.sum(resp, dim=0, keepdim=True).unsqueeze(-1) + eps
        elif self.covariance_type == "diag":
            x2 = (resp * x * x).sum(0, keepdim=True) / pi
            mu2 = mu * mu
            xmu = (resp * mu * x).sum(0, keepdim=True) / pi
            var = x2 - 2 * xmu + mu2 + self.eps

        pi = pi / x.shape[0]

        return pi, mu, var

    def __em(self, x):
        """
        Performs one iteration of the expectation-maximization algorithm by calling the respective subroutines.
        args:
            x:          torch.Tensor (n, 1, d)
        """
        _, log_resp = self._e_step(x)
        pi, mu, var = self._m_step(x, log_resp)

        self.__update_pi(pi)
        self.__update_mu(mu)
        self.__update_var(var)

    def __score(self, x, as_average=True):
        """
        Computes the log-likelihood of the data under the model.
        args:
            x:                  torch.Tensor (n, 1, d)
            sum_data:           bool
        returns:
            score:              torch.Tensor (1)
            (or)
            per_sample_score:   torch.Tensor (n)
        """
        weighted_log_prob = self._estimate_log_prob(x) + torch.log(self.pi)
        per_sample_score = torch.logsumexp(weighted_log_prob, dim=1)

        if as_average:
            return per_sample_score.mean()
        else:
            return torch.squeeze(per_sample_score)

    def __update_mu(self, mu):
        """
        Updates mean to the provided value.
        args:
            mu:         torch.FloatTensor
        """
        assert mu.size() in [(self.n_components, self.n_features), (1, self.n_components, self.n_features)], "Input mu does not have required tensor dimensions (%i, %i) or (1, %i, %i)" % (self.n_components, self.n_features, self.n_components, self.n_features)

        if mu.size() == (self.n_components, self.n_features):
            self.mu = mu.unsqueeze(0)
        elif mu.size() == (1, self.n_components, self.n_features):
            self.mu.data = mu

    def __update_var(self, var):
        """
        Updates variance to the provided value.
        args:
            var:        torch.FloatTensor
        """
        if self.covariance_type == "full":
            assert var.size() in [(self.n_components, self.n_features, self.n_features), (1, self.n_components, self.n_features, self.n_features)], "Input var does not have required tensor dimensions (%i, %i, %i) or (1, %i, %i, %i)" % (self.n_components, self.n_features, self.n_features, self.n_components, self.n_features, self.n_features)

            if var.size() == (self.n_components, self.n_features, self.n_features):
                self.var = var.unsqueeze(0)
            elif var.size() == (1, self.n_components, self.n_features, self.n_features):
                self.var.data = var

        elif self.covariance_type == "diag":
            assert var.size() in [(self.n_components, self.n_features), (1, self.n_components, self.n_features)], "Input var does not have required tensor dimensions (%i, %i) or (1, %i, %i)" % (self.n_components, self.n_features, self.n_components, self.n_features)

            if var.size() == (self.n_components, self.n_features):
                self.var = var.unsqueeze(0)
            elif var.size() == (1, self.n_components, self.n_features):
                self.var.data = var

    def __update_pi(self, pi):
        """
        Updates pi to the provided value.
        args:
            pi:         torch.FloatTensor
        """
        assert pi.size() in [(1, self.n_components, 1)], "Input pi does not have required tensor dimensions (%i, %i, %i)" % (1, self.n_components, 1)

        self.pi.data = pi

    def get_kmeans_mu(self, x, n_centers, init_times=50, min_delta=1e-3):
        """
        Find an initial value for the mean. Requires a threshold min_delta for the k-means algorithm to stop iterating.
        The algorithm is repeated init_times often, after which the best centerpoint is returned.
        args:
            x:            torch.FloatTensor (n, d) or (n, 1, d)
            init_times:   init
            min_delta:    int
        """
        if len(x.size()) == 3:
            x = x.squeeze(1)
        x_min, x_max = x.min(), x.max()
        x = (x - x_min) / (x_max - x_min)

        min_cost = np.inf

        for i in range(init_times):
            tmp_center = x[np.random.choice(np.arange(x.shape[0]), size=n_centers, replace=False), ...]
            l2_dis = torch.norm((x.unsqueeze(1).repeat(1, n_centers, 1) - tmp_center), p=2, dim=2)
            l2_cls = torch.argmin(l2_dis, dim=1)

            cost = 0
            for c in range(n_centers):
                cost += torch.norm(x[l2_cls == c] - tmp_center[c], p=2, dim=1).mean()

            if cost < min_cost:
                min_cost = cost
                center = tmp_center

        delta = np.inf

        while delta > min_delta:
            l2_dis = torch.norm((x.unsqueeze(1).repeat(1, n_centers, 1) - center), p=2, dim=2)
            l2_cls = torch.argmin(l2_dis, dim=1)
            center_old = center.clone()

            for c in range(n_centers):
                center[c] = x[l2_cls == c].mean(dim=0)

            delta = torch.norm((center_old - center), dim=1).max()

        return (center.unsqueeze(0) * (x_max - x_min) + x_min)



def plot_for_GMM_example(data, y):
    n = y.shape[0]

    # fig, ax = plt.subplots(1, 1, figsize=(1.61803398875*4, 4))
    fig, ax = plt.subplots(1, 1)
    ax.set_facecolor("#bbbbbb")
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")

    # ax = fig.add_subplot(1, 3, 2, projection='3d', facecolor='white')
    # ax.scatter(data[:n1, 0], data[:n1, 1], data[:n1, 2], c=sk_pred_label[:n1])
    # ax.set_title(f"S-KM:{(et-st):.1f}ms")

    N,D = data.data.shape
    ### Plot GT: ###
    ax.scatter(data.data[0:N//2,0], data.data[0:N//2,1], color="#000000", s=3, alpha=.75)
    ax.scatter(data.data[N//2:,0], data.data[N//2:,1], color="#ffffff", s=3, alpha=.75)
    ### Plot Predictions: ###
    label_0_logical_mask = (y == 0)
    label_1_logical_mask = (y == 1)
    ax.scatter(data.data[label_0_logical_mask, 0], data.data[label_0_logical_mask, 1], color="#dbe9ff", s=3, alpha=.75)
    ax.scatter(data.data[label_1_logical_mask, 0], data.data[label_1_logical_mask, 1], color="#ffdbdb", s=3, alpha=.75)
    plt.show()

    # # plot the locations of all data points ..
    # for i, point in enumerate(data.data):
    #     if i <= n//2:
    #         # .. separating them by ground truth ..
    #         ax.scatter(*point, color="#000000", s=3, alpha=.75, zorder=n+i)
    #     else:
    #         ax.scatter(*point, color="#ffffff", s=3, alpha=.75, zorder=n+i)
    #
    #     if y[i] == 0:
    #         # .. as well as their predicted class
    #         ax.scatter(*point, zorder=i, color="#dbe9ff", alpha=.6, edgecolors=colors[1])
    #     else:
    #         ax.scatter(*point, zorder=i, color="#ffdbdb", alpha=.6, edgecolors=colors[5])

    handles = [plt.Line2D([0], [0], color="w", lw=4, label="Ground Truth 1"),
        plt.Line2D([0], [0], color="black", lw=4, label="Ground Truth 2"),
        plt.Line2D([0], [0], color=colors[1], lw=4, label="Predicted 1"),
        plt.Line2D([0], [0], color=colors[5], lw=4, label="Predicted 2")]

    legend = ax.legend(loc="best", handles=handles)

    plt.tight_layout()
    # plt.savefig("example.pdf")
    plt.show()

    ### Get Stats: ###


def bla_GMM2_testing():
    n, d = 30000, 2

    # generate some data points ..
    data = torch.Tensor(n, d).normal_()
    # .. and shift them around to non-standard Gaussians
    data[:n // 2] -= 1
    data[:n // 2] *= np.sqrt(3)
    data[n // 2:] += 1
    data[n // 2:] *= np.sqrt(2)

    # Next, the Gaussian mixture is instantiated and ..
    n_components = 2
    model = GaussianMixture(n_components, d)
    model.fit(data)
    # .. used to predict the data points as they where shifted
    y = model.predict(data)


    ### Plot Results: ###
    ### Set Up Plots: ###
    n = y.shape[0]
    N, D = data.data.shape
    # fig, ax = plt.subplots(1, 1, figsize=(1.61803398875*4, 4))
    fig, ax = plt.subplots(1, 1)
    ax.set_facecolor("#bbbbbb")
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    # ### Plot GT: ###
    # ax.scatter(data.data[0:N // 2, 0], data.data[0:N // 2, 1], color="#000000", s=3, alpha=.75)
    # ax.scatter(data.data[N // 2:, 0], data.data[N // 2:, 1], color="#ffffff", s=3, alpha=.75)
    ### Plot Predictions: ###
    label_0_logical_mask = (y == 0)
    label_1_logical_mask = (y == 1)
    ax.scatter(data.data[label_0_logical_mask, 0], data.data[label_0_logical_mask, 1], color="#dbe9ff", s=3, alpha=.75)
    ax.scatter(data.data[label_1_logical_mask, 0], data.data[label_1_logical_mask, 1], color="#ffdbdb", s=3, alpha=.75)
    plt.show()
    ### Plot Ellipses: ###



    # plot_for_GMM_example(data, y)

#
# bla_GMM2_testing()







