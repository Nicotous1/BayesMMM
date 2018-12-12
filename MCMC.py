import numpy as np
from scipy.stats import poisson


def generate_P(alpha):
    # Generate the matrix P with Dirichlet prior of parameter alpha
    m = alpha.shape[0]
    P = np.zeros((m,m))
    for i in range(m):
        P[i] = np.random.dirichlet(alpha[i])
    return P
    
def pick_state(P):
    # Pick a state according to the mass distribution P
    m = len(P)
    return int(np.random.choice(np.arange(m), 1, p = list(P)))

def normalize(a):
    # Normalize a vector (not a matrix)
    a = np.array(a)
    return a/a.sum()

def get_stationary(P):
    # Return the stationary distribution of the markov chain P
    vals, vecs = np.linalg.eig(P.transpose())
    I = np.where(np.round(vals, 2) == 1)
    i = I[0]
    pi_t = vecs[:, i]
    pi_t = pi_t/pi_t.sum()
    return pi_t.transpose()

def get_Dt(Y, lambdas, t):
    # return the Dt of the paper for formula (8)
    m = len(lambdas)
    Dt = np.zeros(m)
    for i in range(m):
        Dt[i] = poisson.pmf(Y[t], lambdas[i])
    return Dt


def iter_MCMC(lambdas, P, Y, alpha, theta):
    # Iterate over formulas (7), (10) and (15) -> MCMC algorithm
    
    m, n = alpha.shape[0], len(Y) # Parameters
    
    # Generate F (2.1)
    F = np.zeros((n,m))
    for t in range(n):
        F_prev = F[t - 1] if t > 0 else get_stationary(P)
        a = np.dot(F_prev, P)
        Dt = get_Dt(Y, lambdas, t)
        F_next = normalize(a*Dt)
        F[t] = F_next
    
    # Generate S and N (2.1 and 2.2)
    S = np.zeros(n)
    N = np.zeros((m,m))
    S[n-1] = pick_state(F[n-1]) # Init
    for t in range(n-2, -1, -1):
        # S
        P_s = normalize(F[t]*P[:, int(S[t+1])])
        S[t] = pick_state(P_s)
        # N
        i, k = int(S[t]), int(S[t+1])
        N[i, k] += 1
        
        
    # Generate P (2.2)
    X = np.random.gamma(alpha + N, 1) # Computing X
    P = X / (X.sum(axis = 1)[:, None]) # Normalizing
    
    
    # Update lambda (method MCMC)
    for k in range(m):
        k_selector = (S == k)
        a = theta[k, 0] + Y[k_selector].sum()
        b = theta[k, 1] + k_selector.sum()
        lambdas[k] = np.random.gamma(a,1/b, 1)
        
    return lambdas, P, S, F


def MCMC(Y, alpha, theta, N, N_rejected = None, lambdas_init = None, P_init = None):
    # Iterate over formulas (7), (10) and (15) -> MCMC algorithm
    # Save the history of parameters after N_rejected simulation
    
    if N_rejected is None: N_rejected = round(0.04*N)
    N_saved = N - N_rejected
    
    # Initialisation via prior
    m = alpha.shape[0]
    lambdas = np.random.gamma(theta[:, 0], 1/theta[:, 1]) if lambdas_init is None else lambdas_init
    P = generate_P(alpha) if P_init is None else P_init
    
    history = np.zeros((N_saved, m, m))
    for i_iter in range(N):
        if (i_iter % 50) == 0: print("Iteration {}/{}...    \r".format(i_iter, N), end = "")
        lambdas, P, S, F = iter_MCMC(lambdas, P, Y, alpha, theta)
        if i_iter >= N_rejected:
            i_saved = i_iter - N_rejected
            history[i_saved] = np.column_stack((lambdas, np.diag(P)))
    
    return F, history