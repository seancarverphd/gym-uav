import numpy as np

class Gaussian2D():
    def __init__(self):
        self.mean_x = None
        self.mean_y = None
        self.var_major = None
        self.var_minor = None
        self.cov_theta = None
        self.var_x_ = None  # trailing underbars mean variable is internal and not under direct control of agents
        self.var_y_ = None
        self.cov_xy_ = None

    def set_mean(self, mean_x, mean_y):
        self.mean_x = mean_x
        self.mean_y = mean_y

    def set_covariance(self, var_major, var_minor, cov_theta):  #TODO this needs to be checked!!
        # "major" axis is actually indicated by max(var_major, var_minor) OK if reversed       
        # How I derived this:
        # The covariance matrix is symmetric and positive definite so its eigen decomposition coincides with this singular decomposition
        # Write down singular values (which in this case are eigenvalues) in a diagonal matrix
        # Write down singular/eigen vector matrices which are rotation in an orthogonal matrix and its transpose.  Then multiply matrices.
        self.var_major = var_major
        self.var_minor = var_minor
        self.cov_theta = cov_theta
        self.var_x_ = var_major*np.sin(cov_theta)**2 + var_minor*np.cos(cov_theta)**2
        self.var_y_ = var_major*np.cos(cov_theta)**2 + var_minor*np.sin(cov_theta)**2
        self.cov_xy_ = (var_major - var_minor)*np.sin(cov_theta)*np.cos(cov_theta)

    def sample(self, size=None):
        return np.random.multivariate_normal(np.array([self.mean_x, self.mean_y]), np.array([[self.var_x_, self.cov_xy_], [self.cov_xy_, self.var_y_]]), size)
