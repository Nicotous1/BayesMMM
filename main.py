import numpy as np
import pandas as pd
import matplotlib.pyplot as plt   

from MCMC import PoissonMixtureMarkov, MCMC
       

# Fetal dataset
Y = pd.read_csv("data/lamb.csv", index_col = 0).values.flatten()

# Priors
alpha = [(3,1), (0.5,0.5)] # Prior to generate P
theta = [(1,2), (2,1)] # Prior to generate lambdas (a,b)

# Init the model
model = PoissonMixtureMarkov(alpha, theta)

# Set the exact priors of the example
lambdas = [0.256, 3.101]
P = [(0.984, 0.016),
     (0.308, 0.692)]
model.set_priors(P, lambdas)

# Running MCMC (take some time)
F, S = MCMC(model, Y, N = 20, N_rejected = 10)


# Computing stats
history = model.get_history()
print(history.describe())

# Plotting final probabilities
n = len(F)    
plt.figure()
plt.subplot(2,1,1)
plt.bar(range(n), F[:, 0])
plt.subplot(2,1,2)
plt.bar(range(n), F[:, 1])


