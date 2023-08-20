
### Import All: ###
from RapidBase.import_all import *

### Sklearn Itself + Utils: ###
import sklearn
from matplotlib.colors import ListedColormap
from matplotlib import colors
from scipy import linalg
from time import time
import warnings
from itertools import cycle, islice

### DataSets: ###
from sklearn.datasets import load_iris, load_digits, load_diabetes, load_boston, load_wine, load_files, make_classification, \
    fetch_20newsgroups, fetch_20newsgroups_vectorized, fetch_california_housing, fetch_covtype, fetch_lfw_pairs,\
    load_breast_cancer, load_linnerud, make_circles, fetch_species_distributions, fetch_olivetti_faces, make_hastie_10_2, make_checkerboard, make_blobs, make_regression, make_moons, make_gaussian_quantiles

### Decomposition: ###
from sklearn.decomposition import PCA, KernelPCA, IncrementalPCA, SparsePCA, TruncatedSVD, FastICA, MiniBatchSparsePCA, NMF, FactorAnalysis

### Preprocessing: ###
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler, OneHotEncoder, Binarizer, LabelEncoder, MultiLabelBinarizer, Normalizer

### Cluster: ###
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering, SpectralCoclustering, MiniBatchKMeans, DBSCAN, OPTICS, FeatureAgglomeration, Birch, SpectralBiclustering
from sklearn.cluster import compute_optics_graph, estimate_bandwidth, ward_tree, kmeans_plusplus

### Pipeline: ###
from sklearn.pipeline import make_pipeline, Bunch, Pipeline

### Manifold: ###
from sklearn.manifold import TSNE, SpectralEmbedding, LocallyLinearEmbedding, Isomap, MDS

### Ensemble: ###
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, ExtraTreesClassifier, StackingClassifier, VotingClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor, StackingRegressor, VotingRegressor, HistGradientBoostingRegressor

### Model Selection: ###
from sklearn.model_selection import KFold, GridSearchCV, StratifiedKFold, RepeatedStratifiedKFold, GroupKFold, StratifiedGroupKFold, RandomizedSearchCV, StratifiedShuffleSplit
from sklearn.model_selection import ParameterGrid, ParameterSampler, PredefinedSplit, RepeatedKFold, ShuffleSplit, train_test_split, validation_curve

### SVM: ###
from sklearn.svm import SVC, LinearSVC, LinearSVR, SVR, OneClassSVM, NuSVC, NuSVR

### Covariance: ###
from sklearn.covariance import ShrunkCovariance, EmpiricalCovariance, EllipticEnvelope, GraphicalLasso, GraphicalLassoCV, OAS

### Trees: ###
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, BaseDecisionTree, ExtraTreeClassifier, ExtraTreeRegressor

### Feature Extraction: ###
from sklearn.feature_extraction import FeatureHasher, DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer, FeatureHasher, HashingVectorizer, TransformerMixin
from sklearn.feature_extraction.image import extract_patches_2d, reconstruct_from_patches_2d, grid_to_graph, img_to_graph, PatchExtractor

### Feature Selection: ###
from sklearn.feature_selection import SelectKBest, SelectFdr, SelectPercentile, SelectFpr, SelectFwe, SelectFromModel, GenericUnivariateSelect, SequentialFeatureSelector
from sklearn.feature_selection import chi2, f_classif, f_oneway, f_regression, mutual_info_classif, mutual_info_regression, r_regression

### Gaussian Processes: ###
from sklearn.gaussian_process import GaussianProcessRegressor, GaussianProcessClassifier
from sklearn.gaussian_process.kernels import Kernel, WhiteKernel, CompoundKernel, ConstantKernel, KernelOperator, PairwiseKernel, GenericKernelMixin, RBF, DotProduct, Exponentiation, ExpSineSquared, Hyperparameter

### Utils: ###
from sklearn.utils import Bunch, shuffle, indices_to_mask, gen_batches, Path, Sequence

### Linear Model: ###
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, LogisticRegressionCV, ARDRegression, RidgeClassifierCV, SGDClassifier, RidgeCV, SGDRegressor, RANSACRegressor, BayesianRidge, RidgeClassifier, \
    ElasticNet, ElasticNetCV, MultiTaskElasticNetCV, MultiTaskLassoCV, MultiTaskElasticNet, Hinge, Huber, HuberRegressor, LassoCV, Lasso, \
    LassoLars, Lars, LarsCV, LassoLarsCV, QuantileRegressor, Perceptron, ridge_regression, SquaredLoss, SGDOneClassSVM

### Metrics: ###
from sklearn.metrics import accuracy_score, f1_score, r2_score, homogeneity_score, silhouette_score, balanced_accuracy_score, roc_auc_score, roc_curve, dcg_score, ndcg_score, \
    fbeta_score, mean_squared_error, mean_absolute_error, classification_report, median_absolute_error, mean_squared_log_error, log_loss, mean_absolute_percentage_error, \
    max_error, confusion_matrix, plot_confusion_matrix, euclidean_distances, nan_euclidean_distances, auc, adjusted_rand_score, adjusted_mutual_info_score, average_precision_score, top_k_accuracy_score, \
    DistanceMetric, hamming_loss, hinge_loss, cohen_kappa_score, brier_score_loss, multilabel_confusion_matrix, precision_score, precision_recall_curve, precision_recall_fscore_support, \
    plot_precision_recall_curve, label_ranking_average_precision_score, label_ranking_loss, plot_det_curve, plot_roc_curve, DetCurveDisplay, RocCurveDisplay, PrecisionRecallDisplay, ConfusionMatrixDisplay, jaccard_score, mutual_info_score
from sklearn.metrics._plot import precision_recall_curve, roc_curve, det_curve, confusion_matrix, base

### Neural Network: ###
from sklearn.neural_network import MLPClassifier, MLPRegressor, BernoulliRBM

### Naive Bayes: ###
from sklearn.naive_bayes import GaussianNB, BernoulliNB, ComplementNB, CategoricalNB, MultinomialNB, LabelBinarizer

### Discriminant Analysis: ###
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis, TransformerMixin, ClassifierMixin, LinearClassifierMixin, StandardScaler

### Neighbors: ###
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor, NearestNeighbors, RadiusNeighborsClassifier, RadiusNeighborsTransformer, \
    RadiusNeighborsRegressor, KNeighborsTransformer, KernelDensity, DistanceMetric, KDTree, BallTree, LocalOutlierFactor, NearestCentroid, NeighborhoodComponentsAnalysis

### Inspection: ###
from sklearn.inspection import PartialDependenceDisplay, DecisionBoundaryDisplay






#### Classification: ###
def classification_tutorial():
    names = [
        "Nearest Neighbors",
        "Linear SVM",
        "RBF SVM",
        "Gaussian Process",
        "Decision Tree",
        "Random Forest",
        "Neural Net",
        "AdaBoost",
        "Naive Bayes",
        "QDA",
    ]

    ### Get All The Classifiers I Want To Check: ###
    classifiers = [
        KNeighborsClassifier(3),
        SVC(kernel="linear", C=0.025),
        SVC(gamma=2, C=1),
        GaussianProcessClassifier(1.0 * RBF(1.0)),
        DecisionTreeClassifier(max_depth=5),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        MLPClassifier(alpha=1, max_iter=1000),
        AdaBoostClassifier(),
        GaussianNB(),
        QuadraticDiscriminantAnalysis(),
    ]

    ### Set Random State: ###
    rng = np.random.RandomState(2)

    ### Get DataSet: ###
    X, y = make_classification(n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=1)
    ### Add Noise: ###
    X += 2 * rng.uniform(size=X.shape)
    linearly_separable = (X, y)

    ### Get More DataSets: ###
    datasets = [
        make_moons(noise=0.3, random_state=0),
        make_circles(noise=0.2, factor=0.5, random_state=1),
        linearly_separable,
    ]
    
    ### Loop Over Datasets And Present Classifier Results: ###
    figure = plt.figure(figsize=(27, 9))
    i = 1
    for dataset_counter, current_dataset in enumerate(datasets):
        ### preprocess dataset, split into training and test part: ###
        X, y = current_dataset
        X = StandardScaler().fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
        
        ### Get Plot Boundaries: ###
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

        ### just plot the dataset first: ###
        cm = plt.cm.RdBu  #colormap
        cm_bright = ListedColormap(["#FF0000", "#0000FF"])
        ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
        if dataset_counter == 0:
            ax.set_title("Input data")
        
        ### Plot the training points: ###
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors="k")
        
        ### Plot the testing points: ###
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6, edgecolors="k")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xticks(())
        ax.set_yticks(())
        
        ### Uptick Counter: ###
        i += 1

        ### Loop Over Classifiers: ###
        for current_classifier_name, current_classifier in zip(names, classifiers):
            ### Fit Current Classifier: ###
            current_classifier.fit(X_train, y_train)
            score = current_classifier.score(X_test, y_test)

            ### Plot Decision Boundaries From Estimator: ###
            ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
            DecisionBoundaryDisplay.from_estimator(current_classifier, X, cmap=cm, alpha=0.8, ax=ax, eps=0.5)

            ### Plot the training points: ###
            ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors="k")

            ### Plot the testing points: ###
            ax.scatter(
                X_test[:, 0],
                X_test[:, 1],
                c=y_test,
                cmap=cm_bright,
                edgecolors="k",
                alpha=0.6,
            )

            ### Set Limits and Add Text: ###
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_xticks(())
            ax.set_yticks(())
            if dataset_counter == 0:
                ax.set_title(current_classifier_name)
            ax.text(
                x_max - 0.3,
                y_min + 0.3,
                ("%.2f" % score).lstrip("0"),
                size=15,
                horizontalalignment="right",
            )

            ### Uptick Counter: ###
            i += 1

    plt.tight_layout()
    plt.show()




def plot_classification_probability():
    ### Load DataSet: ###
    iris = sklearn.datasets.load_iris()
    X = iris.data[:, 0:2]  # we only take the first two features for visualization
    y = iris.target
    number_of_features = X.shape[1]
    
    ### Parameters For Classifiers: ###
    C = 10
    kernel = 1.0 * RBF([1.0, 1.0])  # for GPC

    ### Create different classifiers: ###
    classifiers = {
        "L1 logistic": LogisticRegression(C=C, penalty="l1", solver="saga", multi_class="multinomial", max_iter=10000),
        "L2 logistic (Multinomial)": LogisticRegression(C=C, penalty="l2", solver="saga", multi_class="multinomial", max_iter=10000),  #TODO: understand difference between multi_class and ovr
        "L2 logistic (OvR)": LogisticRegression(C=C, penalty="l2", solver="saga", multi_class="ovr", max_iter=10000),
        "Linear SVC": SVC(kernel="linear", C=C, probability=True, random_state=0),
        "GPC": GaussianProcessClassifier(kernel),
    }
    number_of_classifiers = len(classifiers)


    ### Prepare Subplots Of Probability: ###
    plt.figure(figsize=(3 * 2, number_of_classifiers * 2))
    plt.subplots_adjust(bottom=0.2, top=0.95)
    xx = np.linspace(3, 9, 100)
    yy = np.linspace(1, 5, 100).T
    xx, yy = np.meshgrid(xx, yy)
    XY_grid_points = np.c_[xx.ravel(), yy.ravel()]

    ### Loop Over The Different Classifiers: ###
    for index, (classifier_name, current_classifier) in enumerate(classifiers.items()):
        ### Fit Classifier; ###
        current_classifier.fit(X, y)

        ### Get Predictions And Accuracies: ###
        y_pred = current_classifier.predict(X)
        accuracy = accuracy_score(y, y_pred)
        print("Accuracy (train) for %s: %0.1f%% " % (classifier_name, accuracy * 100))

        ### Get Probabilities: ###
        probas = current_classifier.predict_proba(XY_grid_points)

        ### Loop Over The Different Classises: ###
        n_classes = np.unique(y_pred).size  #get number of classes from y_pred!!!!
        for k in range(n_classes):
            plt.subplot(number_of_classifiers, n_classes, index * n_classes + k + 1)
            plt.title("Class %d" % k)
            if k == 0:
                plt.ylabel(classifier_name)
            imshow_handle = plt.imshow(probas[:, k].reshape((100, 100)), extent=(3, 9, 1, 5), origin="lower")   #NOTICE THE RESHAPE!
            plt.xticks(())
            plt.yticks(())
            idx = y_pred == k
            if idx.any():
                plt.scatter(X[idx, 0], X[idx, 1], marker="o", c="w", edgecolor="k")

    ax = plt.axes([0.15, 0.04, 0.7, 0.05])
    plt.title("Probability")
    plt.colorbar(imshow_handle, cax=ax, orientation="horizontal")

    plt.show()


def recognizing_digits():
    ### Load DataSet: ###
    digits = sklearn.datasets.load_digits()

    ### Show 4 Images From The DataSet: ###
    _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
    for ax, image, label in zip(axes, digits.images, digits.target):
        ax.set_axis_off()
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
        ax.set_title("Training: %i" % label)

    ### Flatten The Images Into Features: ###
    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))

    ### Create a classifier: a support vector classifier: ###
    svc_classifier = SVC(gamma=0.001)

    ### Split data into 50% train and 50% test subsets: ###
    X_train, X_test, y_train, y_test = train_test_split(data, digits.target, test_size=0.5, shuffle=False)

    ### Learn the digits on the train subset: ###
    svc_classifier.fit(X_train, y_train)

    ### Predict the value of the digit on the test subset: ###
    predicted = svc_classifier.predict(X_test)

    ### Show 4 Prediction Results From The Test DataSet: ###
    _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
    for ax, image, prediction in zip(axes, X_test, predicted):
        ax.set_axis_off()
        image = image.reshape(8, 8)
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
        ax.set_title(f"Prediction: {prediction}")

    ### Print Classification Report: ###
    print(
        f"Classification report for classifier {svc_classifier}:\n"
        f"{sklearn.metrics.classification_report(y_test, predicted)}\n"
    )

    ### Display Confusion Matrix(!): ###
    disp = sklearn.metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
    disp.figure_.suptitle("Confusion Matrix")
    print(f"Confusion matrix:\n{disp.confusion_matrix}")

    plt.show()


def linear_and_quadratic_discriminant_analysis_with_covariance_ellipsoid():
    ### Create ColorMap (didn't quite understand what's going on): ###
    cmap = matplotlib.colors.LinearSegmentedColormap(
        "red_blue_classes",
        {
            "red": [(0, 1, 1), (1, 0.7, 0.7)],
            "green": [(0, 0.7, 0.7), (1, 0.7, 0.7)],
            "blue": [(0, 0.7, 0.7), (1, 1, 1)],
        },
    )
    plt.cm.register_cmap(cmap=cmap)

    ### DataSet With Same Covariance Matrix: ###
    def dataset_fixed_cov():
        """Generate 2 Gaussians samples with the same covariance matrix"""
        n, dim = 300, 2
        np.random.seed(0)
        C = np.array([[0.0, -0.23], [0.83, 0.23]])
        X = np.r_[
            np.dot(np.random.randn(n, dim), C),
            np.dot(np.random.randn(n, dim), C) + np.array([1, 1]),
        ]
        y = np.hstack((np.zeros(n), np.ones(n)))
        return X, y

    ### DataSet With Different (transposed) Covariance Matrix: ###
    def dataset_cov():
        """Generate 2 Gaussians samples with different covariance matrices"""
        n, dim = 300, 2
        np.random.seed(0)
        C = np.array([[0.0, -1.0], [2.5, 0.7]]) * 2.0
        X = np.r_[
            np.dot(np.random.randn(n, dim), C),
            np.dot(np.random.randn(n, dim), C.T) + np.array([1, 4]),
        ]
        y = np.hstack((np.zeros(n), np.ones(n)))
        return X, y

    def plot_data(lda, X, y, y_pred, fig_index):
        splot = plt.subplot(2, 2, fig_index)
        if fig_index == 1:
            plt.title("Linear Discriminant Analysis")
            plt.ylabel("Data with\n fixed covariance")
        elif fig_index == 2:
            plt.title("Quadratic Discriminant Analysis")
        elif fig_index == 3:
            plt.ylabel("Data with\n varying covariances")

        ### Get Stats (true-postitive, false-positive) and Relevant Samples: ###
        tp = (y == y_pred)  # True Positive
        (tp0, tp1) = (tp[y == 0], tp[y == 1])
        (X0, X1) = (X[y == 0], X[y == 1])
        (X0_tp, X0_fp) = (X0[tp0], X0[~tp0])
        (X1_tp, X1_fp) = (X1[tp1], X1[~tp1])

        ### class 0: dots: ###
        plt.scatter(X0_tp[:, 0], X0_tp[:, 1], marker=".", color="red")
        plt.scatter(X0_fp[:, 0], X0_fp[:, 1], marker="x", s=20, color="#990000")  # dark red

        ### class 1: dots: ###
        plt.scatter(X1_tp[:, 0], X1_tp[:, 1], marker=".", color="blue")
        plt.scatter(X1_fp[:, 0], X1_fp[:, 1], marker="x", s=20, color="#000099")  # dark blue

        ### class 0 and 1 : areas: ###
        #(1). get meshgrid for prediction/probability
        nx, ny = 200, 100
        x_min, x_max = plt.xlim()
        y_min, y_max = plt.ylim()
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, nx), np.linspace(y_min, y_max, ny))
        #(2). predict probability
        Z = lda.predict_proba(np.c_[xx.ravel(), yy.ravel()])
        Z = Z[:, 1].reshape(xx.shape)
        plt.pcolormesh(xx, yy, Z, cmap="red_blue_classes", norm=matplotlib.colors.Normalize(0.0, 1.0), zorder=0)
        plt.contour(xx, yy, Z, [0.5], linewidths=2.0, colors="white")   #HOW TO CREATE THE WHITE CONTOUR LINE. understand this!!!!!

        ### Plot LDA Means For The Different Classes: ###
        plt.plot(
            lda.means_[0][0],
            lda.means_[0][1],
            "*",
            color="yellow",
            markersize=15,
            markeredgecolor="grey",
        )
        plt.plot(
            lda.means_[1][0],
            lda.means_[1][1],
            "*",
            color="yellow",
            markersize=15,
            markeredgecolor="grey",
        )

        return splot

    ### Plot Ellipse Function: ###
    def plot_ellipse(splot, mean, cov, color):
        v, w = linalg.eigh(cov)
        u = w[0] / linalg.norm(w[0])
        angle = np.arctan(u[1] / u[0])
        angle = 180 * angle / np.pi  # convert to degrees
        # filled Gaussian at 2 standard deviation
        ell = matplotlib.patches.Ellipse(
            mean,
            2 * v[0] ** 0.5,
            2 * v[1] ** 0.5,
            180 + angle,
            facecolor=color,
            edgecolor="black",
            linewidth=2,
        )
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.2)
        splot.add_artist(ell)
        splot.set_xticks(())
        splot.set_yticks(())

    def plot_lda_cov(lda, splot):
        plot_ellipse(splot, lda.means_[0], lda.covariance_, "red")
        plot_ellipse(splot, lda.means_[1], lda.covariance_, "blue")

    def plot_qda_cov(qda, splot):
        plot_ellipse(splot, qda.means_[0], qda.covariance_[0], "red")
        plot_ellipse(splot, qda.means_[1], qda.covariance_[1], "blue")

    ### Plot All Subplots: ###
    plt.figure(figsize=(10, 8), facecolor="white")
    plt.suptitle(
        "Linear Discriminant Analysis vs Quadratic Discriminant Analysis",
        y=0.98,
        fontsize=15,
    )

    ### Plot LDA and QDA for the different datasets: ###
    for i, (X, y) in enumerate([dataset_fixed_cov(), dataset_cov()]):
        ### Linear Discriminant Analysis: ###
        lda = LinearDiscriminantAnalysis(solver="svd", store_covariance=True)
        y_pred = lda.fit(X, y).predict(X)
        splot = plot_data(lda, X, y, y_pred, fig_index=2 * i + 1)
        plot_lda_cov(lda, splot)
        plt.axis("tight")

        # Quadratic Discriminant Analysis
        qda = QuadraticDiscriminantAnalysis(store_covariance=True)
        y_pred = qda.fit(X, y).predict(X)
        splot = plot_data(qda, X, y, y_pred, fig_index=2 * i + 2)
        plot_qda_cov(qda, splot)
        plt.axis("tight")

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.show()



def kmeans_on_digits():
    ### Load Digits DataSet: ###
    data, labels = load_digits(return_X_y=True)
    (n_samples, n_features), n_digits = data.shape, np.unique(labels).size
    print(f"# digits: {n_digits}; # samples: {n_samples}; # features {n_features}")

    def bench_k_means(kmeans, name, data, labels):
        """Benchmark to evaluate the KMeans initialization methods.

        Parameters
        ----------
        kmeans : KMeans instance
            A :class:`~sklearn.cluster.KMeans` instance with the initialization
            already set.
        name : str
            Name given to the strategy. It will be used to show the results in a
            table.
        data : ndarray of shape (n_samples, n_features)
            The data to cluster.
        labels : ndarray of shape (n_samples,)
            The labels used to compute the clustering metrics which requires some
            supervision.
        """
        ### Fit Current KMeans Algoirhtm On Data After Standard Scaling: ###
        t0 = time()
        estimator = make_pipeline(StandardScaler(), kmeans).fit(data)
        fit_time = time() - t0
        results = [name, fit_time, estimator[-1].inertia_]

        ### Define the metrics which require only the true labels and estimator labels (supervised): ###
        clustering_metrics = [
            sklearn.metrics.homogeneity_score,
            sklearn.metrics.completeness_score,
            sklearn.metrics.v_measure_score,
            sklearn.metrics.adjusted_rand_score,
            sklearn.metrics.adjusted_mutual_info_score,
        ]
        results += [current_metric(labels, estimator[-1].labels_) for current_metric in clustering_metrics]

        ### The silhouette score requires the full dataset (unsupervised): ###
        results += [
            sklearn.metrics.silhouette_score(
                data,
                estimator[-1].labels_,
                metric="euclidean",
                sample_size=300,
            )
        ]

        ### Show the results: ###
        formatter_result = (
            "{:9s}\t{:.3f}s\t{:.0f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}"
        )
        print(formatter_result.format(*results))

    print(82 * "_")
    print("init\t\ttime\tinertia\thomo\tcompl\tv-meas\tARI\tAMI\tsilhouette")

    ### K-means++ ###
    kmeans = KMeans(init="k-means++", n_clusters=n_digits, n_init=4, random_state=0)
    bench_k_means(kmeans=kmeans, name="k-means++", data=data, labels=labels)

    ### K-means with random instantiation: ###
    kmeans = KMeans(init="random", n_clusters=n_digits, n_init=4, random_state=0)
    bench_k_means(kmeans=kmeans, name="random", data=data, labels=labels)

    ### PCA -> Kmeans: ###
    pca = PCA(n_components=n_digits).fit(data)
    kmeans = KMeans(init=pca.components_, n_clusters=n_digits, n_init=1)
    bench_k_means(kmeans=kmeans, name="PCA-based", data=data, labels=labels)

    ### PCA -> Kmeans++: ###
    reduced_data = PCA(n_components=2).fit_transform(data)
    kmeans = KMeans(init="k-means++", n_clusters=n_digits, n_init=4)
    kmeans.fit(reduced_data)


    ### Plot the decision boundary. For that, we will assign a color to each: ###
    # Step size of the mesh. Decrease to increase the quality of the VQ.
    h = 0.02  # point in the mesh [x_min, x_max]x[y_min, y_max].
    x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
    y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    ### Obtain labels for each point in mesh. Use last trained model: ###
    #TODO: i think the .ravel() is simply to flatten
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])  #notice the .c_ , understand what's going on

    ### Put the mesghrid result into a color plot: ###
    Z = Z.reshape(xx.shape)  #resgaoe kmeans predictions
    plt.figure(1)
    plt.clf()
    plt.imshow(
        Z,
        interpolation="nearest",
        extent=(xx.min(), xx.max(), yy.min(), yy.max()),
        cmap=plt.cm.Paired,  #TODO: plt.cm.Paired?!?!!?!?
        aspect="auto",
        origin="lower",
    )

    ### Plot/Scatter Actual Data: ###
    plt.plot(reduced_data[:, 0], reduced_data[:, 1], "k.", markersize=2)

    ### Plot the centroids as a white X: ###
    centroids = kmeans.cluster_centers_
    plt.scatter(
        centroids[:, 0],
        centroids[:, 1],
        marker="x",
        s=169,
        linewidths=3,
        color="w",
        zorder=10,
    )
    plt.title(
        "K-means clustering on the digits dataset (PCA-reduced data)\n"
        "Centroids are marked with white cross"
    )
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.show()



def compare_clustering_algorithms():
    1


compare_clustering_algorithms()
# kmeans_on_digits()
# linear_and_quadratic_discriminant_analysis_with_covariance_ellipsoid()
# recognizing_digits()
# plot_classification_probability()
# classification_tutorial()



