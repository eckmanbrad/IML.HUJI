from __future__ import annotations
import numpy as np
from numpy.linalg import inv, det, slogdet

import math


class UnivariateGaussian:
    """
    Class for univariate Gaussian Distribution Estimator
    """
    def __init__(self, biased_var: bool = False) -> UnivariateGaussian:
        """
        Estimator for univariate Gaussian mean and variance parameters

        Parameters
        ----------
        biased_var : bool, default=False
            Should fitted estimator of variance be a biased or unbiased estimator

        Attributes
        ----------
        fitted_ : bool
            Initialized as false indicating current estimator instance has not been fitted.
            To be set as True in `UnivariateGaussian.fit` function.

        mu_: float
            Estimated expectation initialized as None. To be set in `UnivariateGaussian.fit`
            function.

        var_: float
            Estimated variance initialized as None. To be set in `UnivariateGaussian.fit`
            function.
        """

        self.biased_ = biased_var
        self.fitted_, self.mu_, self.var_ = False, None, None

    def fit(self, X: np.ndarray) -> UnivariateGaussian:
        """
        Estimate Gaussian expectation and variance from given samples

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Training data

        Returns
        -------
        self : returns an instance of self.

        Notes
        -----
        Sets `self.mu_`, `self.var_` attributes according to calculated estimation (where
        estimator is either biased or unbiased). Then sets `self.fitted_` attribute to `True`
        """
        # Compute the sample mean and sample variance
        self.mu_ = np.mean(X)
        self.var_ = np.var(X, ddof=1) if not self.biased_ else np.var(X)
        self.fitted_ = True

        return self

    def pdf(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate PDF of observations under Gaussian model with fitted estimators

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Samples to calculate PDF for

        Returns
        -------
        pdfs: ndarray of shape (n_samples, )
            Calculated values of given samples for PDF function of N(mu_, var_)

        Raises
        ------
        ValueError: In case function was called prior fitting the model
        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `pdf` function")

        pdf = np.zeros(len(X))
        for i, x in enumerate(X):
            pdf[i] = UnivariateGaussian.__single_pdf(self.mu_, self.var_, x)
            
        return pdf

    @staticmethod
    def __single_pdf(mu, var, x: float) -> float:
        coefficient1 = -1/(2 * math.pow(var, 2))
        exponent = np.exp(coefficient1 * math.pow(x - mu, 2))
        coefficient2 = 1/math.sqrt(2 * math.pi * math.pow(var, 2))

        return coefficient2 * exponent

    @staticmethod
    def log_likelihood(mu: float, sigma: float, X: np.ndarray) -> float:
        """
        Calculate the log-likelihood of the data under a specified Gaussian model

        Parameters
        ----------
        mu : float
            Expectation of Gaussian
        sigma : float
            Variance of Gaussian
        X : ndarray of shape (n_samples, )
            Samples to calculate log-likelihood with

        Returns
        -------
        log_likelihood: float
            log-likelihood calculated
        """
        s = 0
        for x in X:
            s += (x-mu)**2
        m = len(X)
        return -0.5 * m * np.log(2 * math.pi * sigma) - (s / (2 * sigma))


class MultivariateGaussian:
    """
    Class for multivariate Gaussian Distribution Estimator
    """
    def __init__(self):
        """
        Initialize an instance of multivariate Gaussian estimator

        Attributes
        ----------
        fitted_ : bool
            Initialized as false indicating current estimator instance has not been fitted.
            To be set as True in `MultivariateGaussian.fit` function.

        mu_: ndarray of shape (n_features,)
            Estimated expectation initialized as None. To be set in `MultivariateGaussian.fit`
            function.

        cov_: ndarray of shape (n_features, n_features)
            Estimated covariance initialized as None. To be set in `MultivariateGaussian.fit`
            function.
        """
        self.mu_, self.cov_ = None, None
        self.fitted_ = False

    def fit(self, X: np.ndarray) -> MultivariateGaussian:
        """
        Estimate Gaussian expectation and covariance from given samples

        Parameters
        ----------
        X: ndarray of shape (n_samples, n_features)
            Training data

        Returns
        -------
        self : returns an instance of self

        Notes
        -----
        Sets `self.mu_`, `self.cov_` attributes according to calculated estimation.
        Then sets `self.fitted_` attribute to `True`
        """
        self.mu_ = np.array([x.mean() for x in X.transpose()])
        self.cov_ = np.cov(X.transpose(), ddof=1)

        self.fitted_ = True
        return self

    def pdf(self, X: np.ndarray):
        """
        Calculate PDF of observations under Gaussian model with fitted estimators

        Parameters
        ----------
        X: ndarray of shape (n_samples, n_features)
            Samples to calculate PDF for

        Returns
        -------
        pdfs: ndarray of shape (n_samples, )
            Calculated values of given samples for PDF function of N(mu_, cov_)

        Raises
        ------
        ValueError: In case function was called prior fitting the model
        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `pdf` function")

        pdf = np.zeros(len(X))
        for i, x in enumerate(X):
            pdf[i] = MultivariateGaussian.__single_pdf(self.mu_.T, self.cov_, x)

        return pdf

    @staticmethod
    def __single_pdf(mu, cov, x):
        d = len(mu)
        cov_inverse = np.linalg.inv(cov)
        coefficient = 1/math.sqrt(math.pow(2 * math.pi, d) * np.linalg.det(cov))
        exponent = np.exp(-0.5 * ((x-mu).dot(cov_inverse)).dot((x-mu).T))
        return coefficient * exponent

    @staticmethod
    def log_likelihood(mu: np.ndarray, cov: np.ndarray, X: np.ndarray) -> float:
        """
        Calculate the log-likelihood of the data under a specified Gaussian model

        Parameters
        ----------
        mu : ndarray of shape (n_features,)
            Expectation of Gaussian
        cov : ndarray of shape (n_features, n_features)
            covariance matrix of Gaussian
        X : ndarray of shape (n_samples, n_features)
            Samples to calculate log-likelihood with

        Returns
        -------
        log_likelihood: float
            log-likelihood calculated over all input data and under given parameters of Gaussian
        """
        centered_x = X - mu
        m = len(X)
        d = len(mu)
        return -0.5 * (m * np.log(det(cov) * ((2 * math.pi) ** d)) + (np.dot(centered_x, inv(cov)) * centered_x).sum())
