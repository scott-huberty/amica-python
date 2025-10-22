"""Scikit-learn class wrapper for AMICA."""
import warnings

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, validate_data

from .core import fit_amica

CHECK_ARRAY_KWARGS = {
    "dtype": [np.float64, np.float32],
    "ensure_min_samples": 2,
    "ensure_min_features": 2,
}

class AMICA(TransformerMixin, BaseEstimator):
    """AMICA: adaptive Mixture algorithm for Independent Component Analysis.

    Parameters
    ----------
    n_components : int, default=None
        Number of components to use. If None is passed, all are used.
    n_mixtures : int, default=3
        Number of mixtures components to use in the source model.
        default is 3.
    n_models : int, default=1
        Number of models to learn, only 1 is supported currently.
    batch_size : int, optional
        Batch size for processing data in chunks along the samples axis. If ``None``,
        batching is chosen automatically to keep peak memory under ~1.5 GB, and
        warns if the batch size is below ~8k samples. If the input data is small enough
        to process in one shot, no batching is used. If you want to enforce no batching,
        you can override this memory cap by setting batch_size explicitly, e.g. to
        `X.shape[0]` to process all samples at once.", but note that this may lead to
        high memory usage for large datasets.
    mean_center : bool, default=True
        If True, the data is mean-centered before whitening and fitting.
    whiten : str {"zca", "pca", "variance"}, default="zca"
        whitening strategy.
        - 'zca': Data is whitened and rotated back to original axes. This is equivalent
            to `do_mean=True` + `do_sphere=True` + `do_approx_sphere=False` in the
            Fortran AMICA program.
        - 'pca': Data is whitened and left in the PCA basis. This is equivalent to
            `do_mean=True` + `do_sphere=True` + `do_approx_sphere=True` in the
            Fortran AMICA program.
        - 'variance': Diagonal Normalization. Each feature is scaled by the variance
            across features. This is equivalent to `do_mean=True` + `do_sphere=False` +
            in the Fortran AMICA program.
    max_iter : int, default=500
        Maximum number of iterations during fit.
    tol : float, default=1e-7
        Tolerance for stopping criteria. A positive scalar giving the tolerance at which
        the un-mixing matrix is considered to have converged. The default is 1e-7.
        Whereas the Fortran AMICA program contained tunable tolerance parameters for two
        different stopping criteria ``min_dll`` and ``min_grad_norm``. We only expose
        one parameter, which is applied to both criteria.
    lrate : float, default=0.05
        Initial learning rate for the optimization algorithm. The Fortran AMICA program
        exposed 2 tunable learning rate parameters, ``lrate`` and ``rholrate``, but we
        expose only one for simplicity, which is applied to both.
    pdftype : int, default=0
        Type of source density model to use. Currently only ``0`` is supported,
        which corresponds to the Gaussian Mixture Model (GMM) density.
    do_newton : bool, default=True
        If True, the optimization method will switch to newton updates after
        ``newt_start`` iterations. If ``False``, only SGD updates are used.
    newt_start : int, default=50
        Number of iterations before switching to Newton updates if ``do_newton`` is
        True.
    w_init : ndarray of shape (n_components, n_components), default=None
        Initial un-mixing array. If ``None``, then an array of values drawn from a
        normal distribution is used.
    sbeta_init : ndarray of shape (n_components, n_mixtures), default=None
        Initial scale parameters for the mixture components. If ``None``, then an array
        of values drawn from a uniform distribution is used.
    mu_init : ndarray of shape (n_components, n_mixtures), default=None
        Initial location parameters for the mixture components. If ``None``, then an
        array of values drawn from a normal distribution is used.
    random_state : int, RandomState instance or None, default=None
        Used to initialize ``w_init`` when not specified, with a
        normal distribution. Pass an int, for reproducible results
        across multiple function calls.

    Attributes
    ----------
    components_ : ndarray of shape (n_components, n_features)
        The linear operator to apply to the data to get the independent
        sources. This is equal to the unmixing matrix when ``whiten`` is
        False, and equal to ``np.dot(unmixing_matrix, self.whitening_)`` when
        ``whiten`` is True.
    mixing_ : ndarray of shape (n_features, n_components)
        The pseudo-inverse of ``components_``. It is the linear operator
        that maps independent sources to the data.
    mean_ : ndarray of shape(n_features,)
        The mean over features. Only set if `self.whiten` is True.
    whitening_ : ndarray of shape (n_components, n_features)
        Only set if whiten is 'True'. This is the pre-whitening matrix
        that projects data onto the first `n_components` principal components.
    n_features_in_ : int
        Number of features seen during `fit`.
    n_iter_ : int
        Number of iterations taken to converge during fit.

    Examples
    --------
    >>> from sklearn.datasets import load_digits
    >>> from amica import AMICA
    >>> X, _ = load_digits(return_X_y=True)
    >>> transformer = AMICA(n_components=7, random_state=0)
    >>> X_transformed = transformer.fit_transform(X)
    >>> X_transformed.shape
    (1797, 7)
    """

    def __init__(
            self,
            n_components=None,
            *,
            n_mixtures=3,
            n_models=1,
            mean_center=True,
            whiten="zca",
            max_iter=500,
            tol=1e-7,
            lrate=0.05,
            pdftype=0,
            do_newton=True,
            newt_start=50,
            w_init=None,
            sbeta_init=None,
            mu_init=None,
            random_state=None,
            ):
        super().__init__()
        self.n_components = n_components
        self.n_mixtures = n_mixtures
        self.n_models = n_models
        self.mean_center = mean_center
        self.whiten = whiten
        self._whiten = whiten  # for compatibility
        self.max_iter = max_iter
        self.tol = tol
        self.lrate = lrate
        self.pdftype = pdftype
        self.do_newton = do_newton
        self.newt_start = newt_start
        self.w_init = w_init
        self.sbeta_init = sbeta_init
        self.mu_init = mu_init
        self.random_state = random_state

    def fit(self, X, y=None, verbose=None):
        """Fit the AMICA model to the data X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        y : Ignored
            Not used, present here for API consistency by convention.
        verbose : bool or str or int or None, default=None
            Control verbosity of the logging output. If a str, it can be either
            ``"DEBUG"``, ``"INFO"``, ``"WARNING"``, ``"ERROR"``, or ``"CRITICAL"``.
            Note that these are for convenience and are equivalent to passing in
            ``logging.DEBUG``, etc. For ``bool``, ``True`` is the same as ``"INFO"``,
            ``False`` is the same as ``"WARNING"``. If ``None``, defaults to ``"INFO"``.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        # Validate input data
        X = validate_data(
            self, X=X,
            reset=True,
            **CHECK_ARRAY_KWARGS
            )
        # Fit the model
        fit_dict = fit_amica(
            X,
            n_components=self.n_components,
            n_mixtures=self.n_mixtures,
            n_models=self.n_models,
            mean_center=self.mean_center,
            whiten=self.whiten,
            max_iter=self.max_iter,
            tol=self.tol,
            lrate=self.lrate,
            pdftype=self.pdftype,
            do_newton=self.do_newton,
            newt_start=self.newt_start,
            w_init=self.w_init,
            sbeta_init=self.sbeta_init,
            mu_init=self.mu_init,
            random_state=self.random_state,
            verbose=verbose,
        )

        # Set attributes
        if self.mean_center:
            self.mean_ = fit_dict['mean']
        self.n_features_in_ = X.shape[1]
        self.n_iter_ = np.count_nonzero(fit_dict['LL'])
        self.whitening_ = fit_dict["S"][:self.n_components, :]
        self.mixing_ = fit_dict['A']
        self._unmixing = fit_dict['W'][:, :, 0]
        self.components_ = self._unmixing @ fit_dict['S'][:self.n_components, :]
        return self

    def transform(self, X, copy=True):
        """Recover the sources from X (apply the unmixing matrix).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to transform, where n_samples is the number of samples
            and n_features is the number of features.
        copy : bool, default=True
            If False, data passed to fit can be overwritten. Defaults to True.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_components)
            Estimated sources obtained by transforming the data with the estimated
            unmixing matrix.
        """
        check_is_fitted(self)
        X = validate_data(
            self, X=X, reset=False, copy=copy, dtype=[np.float64, np.float32]
            )
        return X @ self.components_.T

    def fit_transform(self, X, y=None):
        """Fit the model to the data and transform it.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_components)
            Estimated sources obtained by transforming the data with the estimated
            unmixing matrix.
        """
        # Here you would implement the AMICA fitting logic
        X = validate_data(self, X=X, y=None, reset=False)
        n_components = self.n_components
        if self.whiten == "variance" and n_components is not None:
            n_components = None
            warnings.warn("Ignoring n_components with whiten='variance'.")
        # For now, we just call the parent class's method
        self.fit(X)
        return self.transform(X)
