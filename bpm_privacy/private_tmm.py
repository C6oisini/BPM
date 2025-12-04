"""
T Mixture Model (Student's t Mixture Model) with BPM Privacy Protection

This implements the SigmaAlphaCluster algorithm from the MATLAB code,
which is a robust clustering algorithm based on Student's t distribution.
"""

import numpy as np
from scipy.special import digamma  # psi function in MATLAB

from .mechanism import BPM
from .sampling import bpm_sampling


class TMM:
    """
    Student's t Mixture Model (TMM) for clustering.

    This is a robust clustering algorithm that uses Student's t distribution
    instead of Gaussian distribution. It's more resistant to outliers.

    Parameters:
    -----------
    n_components : int
        Number of clusters (default=3)
    nu : float
        Degrees of freedom parameter (default=15)
        If None, it will be estimated from data
    alpha : float
        Scale parameter (default=0.01)
    max_iter : int
        Maximum number of EM iterations (default=500)
    tol : float
        Convergence tolerance (default=0.1)
    convergence_count : int
        Number of times convergence criterion must be met (default=3)
    random_state : int
        Random seed for reproducibility
    """

    def __init__(self, n_components=3, nu=15.0, alpha=0.01,
                 max_iter=500, tol=0.1, convergence_count=3,
                 random_state=None, fixed_nu=False, verbose=False):
        self.n_components = n_components
        self.nu = nu
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.convergence_count = convergence_count
        self.random_state = random_state
        self.fixed_nu = fixed_nu  # Whether to fix nu or estimate it
        self.verbose = verbose

        # Parameters to be learned
        self.means_ = None
        self.tau_ = None  # Posterior probabilities
        self.labels_ = None
        self.n_iter_ = 0
        self.converged_ = False

    def fit(self, X):
        """
        Fit the TMM model to data.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data

        Returns:
        --------
        self
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)

        # X is (n, p), we work with (p, n) like MATLAB code
        x = X.T  # (p, n)
        p, n = x.shape
        g = self.n_components

        # Initialize parameters
        mu = np.zeros((p, g))
        nu = self.nu
        alpha = self.alpha

        # Initialize mu randomly within data range
        low = x.min(axis=1, keepdims=True)  # (p, 1)
        up = x.max(axis=1, keepdims=True)   # (p, 1)
        for i in range(g):
            mu[:, i] = np.random.uniform(low.squeeze(), up.squeeze())

        # Initialize variables
        tau = np.zeros((g, n))
        rho = np.zeros((g, n))
        D = np.zeros((g, n))
        c = (p + nu) / 2
        Q = 0
        dQ = 0
        count = 0

        # EM algorithm
        for itr in range(1, self.max_iter + 1):
            # E-step: Compute distances
            for i in range(g):
                for j in range(n):
                    diff = x[:, j] - mu[:, i]
                    D[i, j] = np.dot(diff, diff)

            # Compute temp = 1 / (1 + D/(nu*alpha))^c
            temp = 1.0 / np.power(1 + D / (nu * alpha), c)  # (g, n)

            # Compute posterior probabilities tau and weights rho
            for i in range(g):
                for j in range(n):
                    tau[i, j] = temp[i, j] / np.sum(temp[:, j])
                    rho[i, j] = (nu + p) / (nu + D[i, j] / alpha)

            # Compute objective function Q
            Q_old = Q
            Q = 0
            for i in range(g):
                for j in range(n):
                    q = tau[i, j] * np.log(1.0 / np.power(1 + D[i, j] / (nu * alpha), c))
                    if not np.isnan(q):
                        Q += q

            # M-step: Update parameters
            # Update mu
            for i in range(g):
                frac_m0 = np.zeros(p)
                frac_m1 = 0
                for j in range(n):
                    frac_m0 += tau[i, j] * x[:, j] / rho[i, j]
                    frac_m1 += tau[i, j] / rho[i, j]
                mu[:, i] = frac_m0 / frac_m1

            # Handle NaN in mu
            mu[np.isnan(mu)] = 0

            # Update alpha
            frac_a0 = 0
            frac_a1 = 0
            for i in range(g):
                for j in range(n):
                    frac_a0 += tau[i, j] * rho[i, j] * D[i, j]
                    frac_a1 += tau[i, j]
            alpha = frac_a0 / frac_a1 / p

            # Update nu (if not fixed)
            if not self.fixed_nu:
                star = 0
                for i in range(g):
                    temp_sum = 0
                    for j in range(n):
                        numerator = nu + p
                        denominator = nu + D[i, j] / alpha
                        temp_sum += tau[i, j] * (np.log(numerator / denominator) - numerator / denominator)
                    star += temp_sum / np.sum(tau[i, :])
                star = star / g

                nu_old = nu
                # Newton's method to solve for nu
                nu = 1.0 / (-1 - star + np.log((nu + p) / 2) - digamma((nu + p) / 2))

                # Refinement step
                delta = 10
                while delta > 0:
                    nu = nu + 0.01
                    delta = -digamma(nu) + np.log(nu) + 1 + star - np.log((nu_old + p) / 2) + digamma((nu_old + p) / 2)

                # Update c with new nu
                c = (p + nu) / 2

            # Print progress
            if self.verbose:
                print(f'Iteration: {itr}, Q: {Q:.4f}, nu: {nu:.4f}, alpha: {alpha:.6f}')

            # Check convergence
            if np.isnan(Q):
                if self.verbose:
                    print("Q became NaN, stopping.")
                break

            if abs(Q_old - Q) < self.tol or abs(dQ - abs(Q_old - Q)) < self.tol:
                count += 1
            else:
                count = 0

            dQ = abs(Q_old - Q)

            if count > self.convergence_count:
                self.converged_ = True
                break

        # Final clustering assignment
        labels = np.argmax(tau, axis=0)

        # Store results
        self.means_ = mu.T  # Convert back to (g, p)
        self.tau_ = tau
        self.labels_ = labels
        self.n_iter_ = itr
        self.nu = nu
        self.alpha = alpha

        return self

    def predict(self, X):
        """
        Predict cluster labels for samples in X.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Samples to predict

        Returns:
        --------
        labels : array, shape (n_samples,)
            Cluster labels
        """
        x = X.T  # (p, n)
        p, n = x.shape
        g = self.n_components
        mu = self.means_.T  # (p, g)

        # Compute distances
        D = np.zeros((g, n))
        for i in range(g):
            for j in range(n):
                diff = x[:, j] - mu[:, i]
                D[i, j] = np.dot(diff, diff)

        # Compute posterior probabilities
        c = (p + self.nu) / 2
        temp = 1.0 / np.power(1 + D / (self.nu * self.alpha), c)

        tau = np.zeros((g, n))
        for i in range(g):
            for j in range(n):
                tau[i, j] = temp[i, j] / np.sum(temp[:, j])

        labels = np.argmax(tau, axis=0)
        return labels

    def predict_proba(self, X):
        """
        Predict posterior probabilities for samples in X.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Samples to predict

        Returns:
        --------
        proba : array, shape (n_samples, n_components)
            Posterior probabilities
        """
        x = X.T  # (p, n)
        p, n = x.shape
        g = self.n_components
        mu = self.means_.T  # (p, g)

        # Compute distances
        D = np.zeros((g, n))
        for i in range(g):
            for j in range(n):
                diff = x[:, j] - mu[:, i]
                D[i, j] = np.dot(diff, diff)

        # Compute posterior probabilities
        c = (p + self.nu) / 2
        temp = 1.0 / np.power(1 + D / (self.nu * self.alpha), c)

        tau = np.zeros((g, n))
        for i in range(g):
            for j in range(n):
                tau[i, j] = temp[i, j] / np.sum(temp[:, j])

        return tau.T  # (n, g)


class PrivateTMM:
    """
    Privacy-preserving Student's t Mixture Model using BPM mechanism.

    This class combines the TMM clustering algorithm with the BPM (Bounded
    Perturbation Mechanism) to provide epsilon-dE privacy protection.

    Parameters:
    -----------
    n_components : int
        Number of clusters
    epsilon : float
        Privacy budget for BPM mechanism
    L : float
        Bound parameter for BPM mechanism
    nu : float
        Degrees of freedom for t-distribution (default=15)
    alpha : float
        Scale parameter (default=0.01)
    max_iter : int
        Maximum number of EM iterations
    tol : float
        Convergence tolerance
    random_state : int
        Random seed
    fixed_nu : bool
        Whether to fix nu or estimate from data
    """

    def __init__(self, n_components=3, epsilon=1.0, L=1.0, nu=15.0,
                 alpha=0.01, max_iter=500, tol=0.1, random_state=None,
                 fixed_nu=False, verbose=False):
        self.n_components = n_components
        self.epsilon = epsilon
        self.L = L
        self.nu = nu
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.fixed_nu = fixed_nu
        self.verbose = verbose

        # Will be set after fitting
        self.means_ = None
        self.labels_ = None
        self.tmm_ = None

    def fit(self, X):
        """
        Fit the private TMM model.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data

        Returns:
        --------
        self
        """
        n_samples, d = X.shape

        # Step 1: User side - BPM perturbation
        if self.verbose:
            print(f"Step 1: Applying BPM perturbation (epsilon={self.epsilon}, L={self.L})")
        bpm = BPM(epsilon=self.epsilon, L=self.L, dimension=d)

        X_perturbed = np.zeros_like(X)
        for i in range(n_samples):
            x_perturbed = bpm_sampling(
                v=X[i],
                k=bpm.k,
                L=bpm.L,
                p_L=bpm.p_L,
                lambda_2r=bpm.lambda_2r
            )
            X_perturbed[i] = x_perturbed

        # Step 2: Server side - fit TMM on perturbed data
        if self.verbose:
            print(f"Step 2: Fitting TMM on perturbed data")
        self.tmm_ = TMM(
            n_components=self.n_components,
            nu=self.nu,
            alpha=self.alpha,
            max_iter=self.max_iter,
            tol=self.tol,
            random_state=self.random_state,
            fixed_nu=self.fixed_nu,
            verbose=self.verbose
        )
        self.tmm_.fit(X_perturbed)

        self.means_ = self.tmm_.means_
        self.labels_ = self.tmm_.labels_

        return self

    def predict(self, X):
        """
        Predict cluster labels for samples in X.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Samples to predict

        Returns:
        --------
        labels : array, shape (n_samples,)
            Cluster labels
        """
        return self.tmm_.predict(X)

    def predict_proba(self, X):
        """
        Predict posterior probabilities for samples in X.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Samples to predict

        Returns:
        --------
        proba : array, shape (n_samples, n_components)
            Posterior probabilities
        """
        return self.tmm_.predict_proba(X)
