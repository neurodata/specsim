import ot
import numpy as np
from sklearn.utils import check_array
from scipy.linalg import orthogonal_procrustes
from sklearn.base import BaseEstimator

class BaseAlign(BaseEstimator):
    def __init__(self):
        pass

    def _check_datasets(self, X, Y):
        # check for numpy-ness
        if not isinstance(X, np.ndarray):
            msg = f"first dataset is a {type(X)}, not an np.ndarray! "
            raise TypeError(msg)
        if not isinstance(Y, np.ndarray):
            msg = f"first dataset is a {type(Y)}, not an np.ndarray! "
            raise TypeError(msg)
        # check for 2-dness and finiteness
        X = check_array(X, copy=True)
        Y = check_array(Y, copy=True)
        # check for equal components
        if X.shape[1] != Y.shape[1]:
            msg = "two datasets have different number of components!"
            raise ValueError(msg)
        return X, Y

    def fit(self, X, Y):
        pass

    def transform(self, X):
        if not isinstance(X, np.ndarray):
            msg = f"dataset is a {type(X)}, not an np.ndarray! "
            raise TypeError(msg)
        X = check_array(X)
        return X @ self.Q_

    def fit_transform(self, X, Y):
        self.fit(X, Y)
        return self.transform(X)

class SignFlips(BaseAlign):
    def __init__(
        self,
        criteria="median",
    ):
        # checking criteria argument
        if type(criteria) is not str:
            raise TypeError("criteria must be str")
        if criteria not in ["median", "max"]:
            raise ValueError("{} is not a valid criteria.".format(criteria))

        super().__init__()

        self.criteria = criteria

    def set_criteria_function(self):
        # perform a check, in case it was modified directly
        if self.criteria not in ["median", "max"]:
            raise ValueError("{} is not a valid criteria".format(self.criteria))

        if self.criteria == "median":

            def median_criteria(X):
                return np.median(X, axis=0)

            self.criteria_function_ = median_criteria
        if self.criteria == "max":

            def max_criteria(X):
                return X[np.argmax(np.abs(X), axis=0), np.arange(X.shape[1])]

            self.criteria_function_ = max_criteria

    def fit(self, X, Y):
        X, Y = self._check_datasets(X, Y)
        _, d = X.shape

        self.set_criteria_function()
        X_criterias = self.criteria_function_(X)
        Y_criterias = self.criteria_function_(Y)

        val = np.multiply(X_criterias, Y_criterias)
        t_X = (val > 0) * 2 - 1

        self.Q_ = np.diag(t_X)
        return self

class OrthogonalProcrustes(BaseAlign):
    def __init__(
        self,
        ):

        super().__init__()

    def fit(self, X, Y):
        X, Y = self._check_datasets(X, Y)

        _, d = X.shape
        if X.shape[0] != Y.shape[0]:
            msg = (
                "two datasets have different number of entries! "
                "OrthogonalProcrustes assumes that entries of the two "
                "datasets are matched. consider using SeedlessProcrustes "
                "instead."
            )
            raise ValueError(msg)

        _, d = X.shape
        self.Q_, _ = orthogonal_procrustes(X, Y)
        return self

class SeedlessProcrustes(BaseAlign):
    """
    Implements an algorithm that matches two datasets using an orthogonal
    matrix. Unlike OrthogonalProcrustes, this does not use a matching between
    entries. It can even be used in the settings when the two datasets do not
    have the same number of entries.
    In graph setting, it is used to align the embeddings of two different
    graphs, when it requires some simultaneous inference task, for example,
    inside of the test for the equivalence of the latent distributions.
    In essence, it aims to simultaneously obtain a, not necessarily 1-to-1,
    correspondance between the vertices of the two data sets, and the
    orthogonal matrix alignment matrix. It does via a sequence of alternating
    steps, similarly to a regular expectation-maximization procedures. The step
    of obtaining an assignment ("E-step") is done by solving an optimal
    transport problem via Sinkhorn algorithm, whereas obtaining an orthogonal
    alignment matrix ("M-step") is done via regular orthogonal procurstes.
    Parameters
    ----------
        optimal_transport_lambda : float (default=0.1), optional
            Regularization term of the Sinkhorn optimal transport algorithm.
        optimal_transport_eps : float (default=0.01), optional
            Tolerance parameter for the each Sinkhorn optimal transport
            algorithm. I.e. tolerance for each "E-step".
        optimal_transport_num_reps : int (default=1000), optional
            Number of repetitions in each iteration of the iterative optimal
            transport problem. I.e. maximum number of repetitions in each
            "E-step".
        iterative_eps : float (default=0.1), optional
            Tolerance for the each iteration of the iterative optimal transport
            problem. I.e. tolerance of the whole "EM" algorithm.
        iterative_num_reps : int (default=100), optional
            Number of reps in each iteration of the iterative optimal transport
            problem. I.e. maxumum number of total iterations the whole "EM"
            algorithm.
        init : string, {"2d" (default), "sign_flips", "custom"}, optional
            - "2d"
                uses 2^d different initiazlizations, where d is the dimension.
                specifically, uses all possible matrices with all entries real
                and diagonal entries having magnitude 1 and 0s everywehre else.
                for example, for d=2, tries [[1, 0], [0, 1]], [[1, 0], [0,
                -1]], [[-1, 0], [0, 1]], and [[-1, 0], [0, -1]]. picks the best
                one based on the value of the objective function.
            - 'sign_flips'
                for the first dimension, if two datasets have medians with
                varying signs, flips all signs along this dimension for the
                first dataset. then initializes to an identity.
            - "custom"
                expects either an initial matrix Q or initial matrix P during
                the use of fit or fit_transform. uses initial Q provided,
                unless it is not provided. if not provided - uses initial P. if
                neither is given initializes to Q = I.
        initial_Q : np.ndarray, shape (d, d) or None, optional (default=None)
            An initial guess for the alignment matrix, if such exists. Ignored
            if init alignment is set to anything other than 'custom'.
            If None - initializes using an initial guess for P.
            If None, and P is also None - initializes Q to identity matrix.
        initial_P : np.ndarray, shape (n, m) or None, optional (default=None)
            Initial guess for the initial transport matrix.
            Only matters if Q=None.
    Attributes
    ----------
        Q_ : array, size (d, d)
              final orthogonal matrix, used to modify X.
        P_ : array, size (n, m) where n and md are the sizes of two datasets
             final matrix of optimal transports
    References
    ----------
    .. [1] Agterberg, J.
    """

    def __init__(
        self,
        optimal_transport_lambda=0.1,
        optimal_transport_eps=0.01,
        optimal_transport_num_reps=1000,
        iterative_eps=0.01,
        iterative_num_reps=100,
        init="2d",
        initial_Q=None,
        initial_P=None,
    ):
        # check optimal_transport_lambda argument
        if type(optimal_transport_lambda) is not float:
            msg = "optimal_transport_lambda must be a float, not {}".format(
                type(optimal_transport_lambda)
            )
            raise TypeError(msg)
        if optimal_transport_lambda <= 0:
            msg = "{} is an invalud value of the optimal transport lambda, must be non-negative".format(
                optimal_transport_lambda
            )
            raise ValueError(msg)
        # check optimal_transport_lambda argument
        if type(optimal_transport_eps) is not float:
            msg = "optimal_transport_eps must be a float, not {}".format(
                type(optimal_transport_eps)
            )
            raise TypeError(msg)
        if optimal_transport_eps <= 0:
            msg = "{} is an invalud value of the optimal transport eps, must be postitive".format(
                optimal_transport_eps
            )
            raise ValueError(msg)
        # check optimal_transport_num_reps argument
        if type(optimal_transport_num_reps) is not int:
            msg = "optimal_transport_num_reps must be a int, not {}".format(
                type(optimal_transport_num_reps)
            )
            raise TypeError(msg)
        if optimal_transport_num_reps < 1:
            msg = "{} is invalid number of repetitions, must be greater than 1".format(
                iterative_num_reps
            )
            raise ValueError(msg)
        # check iterative_eps argument
        if type(iterative_eps) is not float:
            msg = "iterative_eps must be a float, not {}".format(type(iterative_eps))
            raise TypeError(msg)
        if iterative_eps <= 0:
            msg = (
                "{} is an invalud value of the iterative eps, must be postitive".format(
                    iterative_eps
                )
            )
            raise ValueError(msg)
        # check iterative_num_reps argument
        if type(iterative_num_reps) is not int:
            msg = "iterative_num_reps must be a int, not {}".format(
                type(iterative_num_reps)
            )
            raise TypeError(msg)
        if iterative_num_reps < 1:
            msg = "{} is invalid number of repetitions, must be greater than 1".format(
                iterative_num_reps
            )
            raise ValueError(msg)
        # check init argument
        if type(init) is not str:
            msg = "initalization must be a str, not {}".format(type(init))
            raise TypeError(msg)
        inits_supported = ["2d", "sign_flips", "custom"]
        if init not in inits_supported:
            msg = "supported inits are {}".format(inits_supported)
            raise ValueError(msg)
        # check initial_Q argument
        if initial_Q is not None:
            if not isinstance(initial_Q, np.ndarray):
                msg = f"initial_Q must be np.ndarray or None, not {type(initial_Q)}"
                raise TypeError(msg)
            initial_Q = check_array(initial_Q, copy=True)
            if initial_Q.shape[0] != initial_Q.shape[1]:
                msg = "initial_Q must be a square orthogonal matrix"
                raise ValueError(msg)
            if not np.allclose(initial_Q.T @ initial_Q, np.eye(initial_Q.shape[0])):
                msg = "initial_Q must be a square orthogonal matrix"
                raise ValueError(msg)
        # check initial_P argument
        if initial_P is not None:
            if not isinstance(initial_P, np.ndarray):
                msg = f"initial_P must be np.ndarray or None, not {type(initial_P)}"
                raise TypeError(msg)
            initial_P = check_array(initial_P, copy=True)
            n, m = initial_P.shape
            if not (
                np.allclose(initial_P.sum(axis=0), np.ones(m) / m)
                and np.allclose(initial_P.sum(axis=1), np.ones(n) / n)
            ):
                msg = (
                    "initial_P must be a doubly stochastic matrix "
                    "(rows add up to (1/number of cols) "
                    "and columns add up to (1/number of rows))"
                )
                raise ValueError(msg)

        super().__init__()

        self.optimal_transport_eps = optimal_transport_eps
        self.optimal_transport_num_reps = optimal_transport_num_reps
        self.optimal_transport_lambda = optimal_transport_lambda
        self.iterative_eps = iterative_eps
        self.iterative_num_reps = iterative_num_reps
        self.init = init
        self.initial_Q = initial_Q
        self.initial_P = initial_P

    def _orthogonal_matrix_from_int(self, val_int, d):
        val_bin = bin(val_int)[2:]
        val_bin = "0" * (d - len(val_bin)) + val_bin
        return np.diag(np.array([(float(i) - 0.5) * -2 for i in val_bin]))

    def _optimal_transport(self, X, Y, Q):
        # "E step" of the SeedlessProcrustes.
        n, d = X.shape
        m, _ = Y.shape
        # initialize probability mass arrays & the cost matrix ; run sinkhorn
        probability_mass_X = np.ones(n) / n
        probability_mass_Y = np.ones(m) / m
        cost_matrix = (
            np.linalg.norm((X @ Q).reshape(n, 1, d) - Y.reshape(1, m, d), axis=2) ** 2
        )
        P = ot.sinkhorn(
            a=probability_mass_X,
            b=probability_mass_Y,
            M=cost_matrix,
            reg=self.optimal_transport_lambda,
            numItermax=self.optimal_transport_eps,
            stopThr=self.optimal_transport_eps,
        )
        return P

    def _procrustes(self, X, Y, P_i):
        # "M step" of the SeedlessProcurstes.
        aligner = OrthogonalProcrustes()
        Q = aligner.fit(X, P_i @ Y).Q_
        return Q

    def _iterative_ot(self, X, Y, Q):
        for i in range(self.iterative_num_reps):
            P_i = self._optimal_transport(X, Y, Q)
            Q = self._procrustes(X, Y, P_i)
            c = np.linalg.norm(X @ Q - P_i @ Y, ord="fro")
            if c < self.iterative_eps:
                break
        return P_i, Q

    def fit(self, X, Y):
        """
        Uses the two datasets to learn the matrix Q_ that aligns the first
        dataset with the second.
        Parameters
        ----------
        X : np.ndarray, shape (n, d)
            First dataset of vectors. These vectors need to have same number of
            dimensions as ones in Y, but the number of vectors can differ.
        Y : np.ndarray, shape (m, d)
            Second dataset of vectors. These vectors need to have same number
            of dimensions as ones in X, but the number of vectors can differ.
        Returns
        -------
        self : returns an instance of self
        """
        X, Y = self._check_datasets(X, Y)
        _, d = X.shape

        if self.init == "2d":
            n, d = X.shape
            m, _ = Y.shape
            P_matrices = np.zeros((2 ** d, n, m))
            Q_matrices = np.zeros((2 ** d, d, d))
            objectives = np.zeros(2 ** d)
            # try 2^d different initializations
            for i in range(2 ** d):
                initial_Q = self._orthogonal_matrix_from_int(i, d)
                P_matrices[i], Q_matrices[i] = P, Q = self._iterative_ot(
                    X, Y, initial_Q
                )
                objectives[i] = np.linalg.norm(X @ Q - P @ Y, ord="fro")
            # pick the best one, using the objective function value
            best = np.argmin(objectives)
            self.initial_Q = self._orthogonal_matrix_from_int(best, d)
            self.P_, self.Q_ = P_matrices[best], Q_matrices[best]
        elif self.init == "sign_flips":
            aligner = SignFlips()
            self.initial_Q = aligner.fit(X, Y).Q_
            self.P_, self.Q_ = self._iterative_ot(X, Y, self.initial_Q)
        else:
            # determine initial Q if "custom" and not provided
            if self.initial_Q is None:
                if self.initial_P is not None:
                    # use initial P, if provided
                    self.initial_Q = self._procrustes(X, Y, self.initial_P)
                else:
                    # set to initial Q to identity if neither Q nor P provided
                    self.initial_Q = np.eye(X.shape[1])
            self.P_, self.Q_ = self._iterative_ot(X, Y, self.initial_Q)

        return self


    

