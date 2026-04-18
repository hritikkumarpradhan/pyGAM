import numpy as np


class EDoFBenchmark:
    """
    Benchmarks the memory and runtime of the Effective Degrees of Freedom
    calculation, specifically targeting the O(N^3) bottleneck in overparameterized models.
    """

    timeout = 120

    def setup(self):
        # Generate an overparameterized U1 matrix (p >> n)
        self.n_samples = 500
        self.n_features = 10000
        self.U1 = np.random.rand(self.n_features, self.n_samples)

    def time_legacy_edof(self):
        # Times the historical bottleneck
        return np.diagonal(self.U1.dot(self.U1.T))

    def peakmem_legacy_edof(self):
        # Tracks the peak RAM allocation of the historical bottleneck
        return np.diagonal(self.U1.dot(self.U1.T))
