import numpy as np
import pandas as pd
from scipy.stats import poisson

class MarkovChain(object):
    #
    # This class is just an abstract class that should be inherited by sub models (fetal, GDP..)
    # It contains all the general function to compute an MCMC
    #
    
    def __init__(self, alpha, theta):
        # Base parameters
        self.alpha = np.array(alpha) # Priors for the dirichlet of P
        self.theta = np.array(theta) # Priors for thetha (depends of the child model)
        
        # Computing parameters
        self.history = [] # Stock the history of parameters (usefull for stats)
        self.has_priors = False # True if the priors has been generated
        self._load_params() # Compute some shortcut
    
    def _load_params(self):
        self.m = len(self.alpha)
        
    def generate_priors(self):
        self.generate_P()
        self.generate_theta()
        self.has_priors = True
    
    def generate_P(self, N = None):
        m = self.m
        N = np.zeros((m,m)) if N is None else N
        X = np.random.gamma(self.alpha + N, 1) # Computing X
        self.P = X / (X.sum(axis = 1)[:, None]) # Normalizing
        return self.P
    
    def get_stationary(self):
        # Return the stationary distribution of the markov chain P
        vals, vecs = np.linalg.eig(self.P.transpose())
        I = np.where(np.round(vals, 2) == 1)
        i = I[0]
        pi_t = vecs[:, i]
        pi_t = pi_t/pi_t.sum()
        return pi_t.transpose()

    def get_Dt(self, Y, t):
        # return the Dt of the paper for formula (8)
        m = self.m
        Dt = np.zeros(m)
        for i in range(m):
            Dt[i] = self.f(Y, t, i)
        return Dt
    
    def get_history(self):
        return pd.DataFrame(self.history, columns = self.get_params(name = True))
    
    def save(self):
        params = self.get_params()
        self.history.append(params)
    
    
    #
    # This function should be implemented by the child model to run the MCMC
    #
    def f(self, Y, t, st):
        raise ValueError("This function should be implemented by the child class !")
    
    def generate_theta(self):
        raise ValueError("This function should be implemented by the child class !")
    
    def generate_theta_MCMC(self, Y = None, S = None):
        raise ValueError("This function should be implemented by the child class !")
    
    def get_params(self, name = False):
        raise ValueError("This function should be implemented by the child class !")
        
        
        
        
        
class PoissonMixtureMarkov(MarkovChain):        
        
    def f(self, Y, t, st):
        return poisson.pmf(Y[t], self.lambdas[st])
    
    def generate_theta(self):
        self.generate_theta_MCMC(Y = [], S = [])
    
    def generate_theta_MCMC(self, Y, S):
        Y, S = np.array(Y), np.array(S)
        if len(S) != len(Y): raise ValueError("S and Y should have the same length to update theta !")
        
        # Update lambda (method MCMC)
        self.lambdas = np.zeros(self.m)
        for k in range(self.m):
            k_selector = (S == k)
            a = self.theta[k, 0] + Y[k_selector].sum()
            b = self.theta[k, 1] + k_selector.sum()
            self.lambdas[k] = np.random.gamma(a,1/b, 1)
    
    def get_params(self, name = False):
        if name:
            return ["lambda{}".format(i) for i in range(self.P.shape[0])] + ["p{}".format(i) for i in range(self.P.shape[0])] # Name of the parameters
        else:
            return np.column_stack((self.lambdas, np.diag(self.P))).flatten() # Parameters values
    
    def set_priors(self, P, lambdas):
        self.P = np.array(P)
        self.lambdas = np.array(lambdas)
        self.has_priors = True
    
    
    
    
        
    
def pick_state(P):
    # Pick a state according to the mass distribution P
    m = len(P)
    return int(np.random.choice(np.arange(m), 1, p = list(P)))

def normalize(a):
    # Normalize a vector (not a matrix)
    a = np.array(a)
    return a/a.sum()


def iter_MCMC(model, Y):
    # Iterate over formulas (7), (10) and (15) -> MCMC algorithm
    
    m, n = model.m, len(Y) # Parameters
    
    # Generate F (2.1)
    F = np.zeros((n,m))
    for t in range(n):
        F_prev = F[t - 1] if t > 0 else model.get_stationary()
        a = np.dot(F_prev, model.P)
        Dt = model.get_Dt(Y, t)
        F_next = normalize(a*Dt)
        F[t] = F_next
    
    # Generate S and N (2.1 and 2.2)
    S = np.zeros(n)
    N = np.zeros((m,m))
    S[n-1] = pick_state(F[n-1]) # Init
    for t in range(n-2, -1, -1):
        # S
        P_s = normalize(F[t]*model.P[:, int(S[t+1])])
        S[t] = pick_state(P_s)
        # N
        i, k = int(S[t]), int(S[t+1])
        N[i, k] += 1
        
        
    # Generate P (2.2)
    model.generate_P(N)
    
    # Update lambda (method MCMC)
    model.generate_theta_MCMC(Y, S)
        
    return F, S


def MCMC(model, Y, N, N_rejected = None, priors = None):
    # Iterate over formulas (7), (10) and (15) -> MCMC algorithm
    # Save the history of parameters after N_rejected simulation
    
    if N_rejected is None: N_rejected = round(0.04*N)
    
    # Initialisation via prior
    if not(model.has_priors):
        model.generate_priors()
    
    for i_iter in range(N):
        if (i_iter % 50) == 0: print("Iteration {}/{}...    \r".format(i_iter, N), end = "")
        F, S = iter_MCMC(model, Y)
        if i_iter >= N_rejected:
            model.save()
    
    print("MCMC has finished to iterate !")
    return F, S