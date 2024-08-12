import numpy as np
from numpy import log
from numpy.linalg import inv
from scipy import stats
import tqdm
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from ARS import ARS

##############################################################################################
#
#   Author: Ani Bhalekar 
#   Date: 5/08/2024 
#   
#   Purpose: 
#           Application of Dirichlet Process Mixture Model - follows 
#           through from Rasmussen's original paper (1999) about iGMMs
#           but for the multivariate case
#
##############################################################################################

class DPMM:
    def __init__(self, alpha=1.0, init_components=1, sample_alpha = False):
        self.alpha = alpha
        self.components = init_components
        self.means = None
        self.covariances = None
        self.precisions = None
        self.weights = None
        self.is_sample_alpha = sample_alpha

        
    def init_priors(self, X): 
        # Hyperpriors and parameter initialisation
        self.cov_y = np.cov(X.T)
        self.mean_y = np.mean(X, axis=0)
        self.W = self.cov_y
        self.beta = 0
        self.xi = self.mean_y
        try:
            self.R = inv(self.W)
        except np.linalg.LinAlgError:
            self.R = np.eye(len(X[0]))
    
    def update_priors(self): 
        try: 
            new_W = (inv(self.cov_y) + self.beta * np.sum(self.precisions, axis = (0,1)))
            self.W = new_W
        except np.linalg.LinAlgError: 
            self.W = self.W

    def log_posterior_log_alpha(self, alpha, k, n):
            log_alpha = np.exp(alpha)
            n = min(n, 1000)
            t1 = (k-2.5) * log(log_alpha)
            t2 = -1/(2*log_alpha)
            t3 = 0.5 * log(2*np.pi*log_alpha) + log_alpha *log(log_alpha/np.e)
            t4 = 0.5 * log(2*np.pi*(log_alpha + n)) + (log_alpha + n) *log((log_alpha + n)/np.e)

            return t1 + t2 + t3 - t4
    
    def sample_alpha(self): 
        # Log posterior distribution of alpha posterior
        # We use the stirling approximation for the Gamma function
        # NOT RECOMMENDED DUE TO NUMERICAL INSTABILITY 
        try: 
            ars = ARS(self.log_posterior_log_alpha, None, xi = [-7, 0, 5], k = self.components, n = self.n)
            self.alpha = np.exp(ars.draw(10)[-1])
        except: 
            return 

    def fit(self, X, n_iterations=100):

        if len(X) == 1: 
            self.means = [X[0]]
            self.covariances = [np.eye(1)]
            self.precisions = [inv(cov) for cov in self.covariances]
            self.weights = [1.0]
            assignments = np.zeros(1, dtype=int)

            return 
        try:
            n_samples, n_features = X.shape
        except: 
            print(X)
            raise ValueError
        
        self.n = n_samples
        # Initialise priors
        self.init_priors(X)

        # Initialize parameters
        self.means = [X[np.random.choice(n_samples)]]
        self.covariances = [np.eye(n_features)]
        self.precisions = [inv(cov) for cov in self.covariances]
        self.weights = [1.0]

        assignments = np.zeros(n_samples, dtype=int)
        
        for _ in range(n_iterations):
            # Sample assignments
            i = 0
            for dummy in tqdm.tqdm(range(n_samples)):
                log_probs = []
                for j in range(self.components):
                    try:
                        log_prob = np.log(self.weights[j]) + stats.multivariate_normal.logpdf(
                            X[i], self.means[j], self.covariances[j], allow_singular=False)
                    except ValueError:
                        log_prob = -np.inf
                    log_probs.append(log_prob)
                
                # Consider a new component
                log_probs.append(np.log(self.alpha) + 
                                 stats.multivariate_normal.logpdf(X[i], np.zeros(n_features), np.eye(n_features)))
                
                log_probs = np.array(log_probs)
                log_probs -= np.max(log_probs)  # For numerical stability
                probs = np.exp(log_probs)
                probs /= np.sum(probs)
                
                new_assignment = np.random.choice(self.components + 1, p=probs)
                               
                if new_assignment == self.components:
                    self.components += 1
                    self.means.append(X[i])
                    self.covariances.append(np.eye(n_features))
                    self.precisions.append(np.eye(n_features))
                    self.weights.append(1.0)
                
                assignments[i] = new_assignment
                self.update_priors()
                i += 1
            
            # Update parameters
            counts = np.bincount(assignments, minlength=self.components)
            self.weights = list((counts + self.alpha / self.components) / (n_samples + self.alpha))
            
            # Keep track of components to remove
            components_to_remove = []
            for j in range(self.components):
                cluster_points = X[assignments == j]
                y_j_bar = np.mean(cluster_points, axis = 0)
                if len(cluster_points) > 1:
                    self.means[j] = y_j_bar
                    cov = np.cov(cluster_points.T, ddof=1)  # Use ddof=1 for sample covariance
                    if np.isscalar(cov):
                        cov = np.array([[cov]])
                    # Ensure the covariance matrix is positive definite
                    cov = cov + 1e-6 * np.eye(n_features) + self.W * self.beta
                    self.covariances[j] = cov
                    self.precisions[j] = inv(cov)
                elif len(cluster_points) == 1:
                    self.means[j] = cluster_points[0]
                    self.covariances[j] = np.eye(n_features) 
                    self.precisions[j] = np.eye(n_features) 
                else:
                    components_to_remove.append(j)
            
            # Remove empty clusters
            for j in sorted(components_to_remove, reverse=True):
                del self.means[j]
                del self.covariances[j]
                del self.precisions[j]
                del self.weights[j]
                assignments[assignments > j] -= 1
            
            self.components -= len(components_to_remove)
            
            # Ensure at least one component remains
            if self.components == 0:
                self.components = 1
                self.means = [X[np.random.choice(n_samples)]]
                self.covariances = [np.eye(n_features)]
                self.precisions = [np.eye(n_features)]
                self.weights = [1.0]
                assignments = np.zeros(n_samples, dtype=int)

            # Sample from alpha
            if self.is_sample_alpha: self.sample_alpha()
        return assignments
    
    def predict(self, X):
        n_samples = X.shape[0]
        assignments = np.zeros(n_samples, dtype=int)
        
        for i in range(n_samples):
            log_probs = []
            for j in range(self.components):
                try:
                    log_prob = np.log(self.weights[j]) + stats.multivariate_normal.logpdf(
                        X[i], self.means[j], self.covariances[j], allow_singular=False)
                except ValueError:
                    log_prob = -np.inf
                log_probs.append(log_prob)
            
            assignments[i] = np.argmax(log_probs)
        
        return assignments
    
    def return_log_probs(self, X): 
        n_samples = X.shape[0]
        max_log_probs = []
        
        for i in range(n_samples):
            log_probs = []
            for j in range(self.components):
                try:
                    log_prob = np.log(self.weights[j]) + stats.multivariate_normal.logpdf(
                        X[i], self.means[j], self.covariances[j], allow_singular=False)
                except ValueError:
                    log_prob = -np.inf

                log_probs.append(log_prob)
            max_log_probs.append(np.max(log_probs))
    
        return max_log_probs


def confidence_ellipse(mean, cov, ax, n_std=3.0, facecolor='none', **kwargs):

    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        facecolor=facecolor,
        **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = mean[0]

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = mean[1]

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

# Example usage
if __name__ == "__main__":
    # Generate some sample data
    np.random.seed(0)
    syn_data = []
    for i in range(50): 
        syn_data.append([np.random.normal(2,0.5), np.random.normal(2,0.5)])

    for i in range(50): 
        syn_data.append(np.array(np.random.multivariate_normal([-16,-16], [[2,-0.5],[-0.5,5]])))

    for i in range(50): 
        syn_data.append(np.array(np.random.multivariate_normal([4, -2], [[1,-1],[-1,2]])))
    
    for i in range(100): 
        syn_data.append(np.array(np.random.multivariate_normal([0,0], [[0.4,-0.2],[-0.2,0.1]])))

    for i in range(100): 
        syn_data.append(np.array(np.random.multivariate_normal([10,10], [[4,-0.2],[-0.2,0.1]])))


    X = np.array(syn_data) * 1

    # Create and fit the DPMM
    dpmm = DPMM(alpha=1e-4)
    assignments = dpmm.fit(X, n_iterations=20)

    print(f"Number of clusters found: {dpmm.components}")
    print(f"Cluster assignments: {assignments}")

    # Make predictions on new data
    X_new = X[:10]
    predictions = dpmm.predict(X_new)
    print(f"Predictions for new data: {predictions}")

    fig, ax = plt.subplots()
    #ax = fig.add_subplot(111, projection="3d", elev=48, azim=134)
    plt.cla()
    ax.scatter(X[:, 0], X[:, 1], c = assignments, cmap=plt.cm.nipy_spectral, edgecolor="k")
    for mean, cov in zip(dpmm.means, dpmm.covariances): 
        confidence_ellipse(mean, cov, ax, 3, "none", edgecolor = "red")
    plt.show()
