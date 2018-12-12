import numpy as np
import pandas as pd
import matplotlib.pyplot as plt   

from MCMC import FetalChain, MCMC
       

# Fetal dataset
Y = pd.read_csv("data/lamb.csv", index_col = 0).values.flatten()

# Priors
alpha = np.array([(3,1), (0.5,0.5)]) # Prior to generate P
theta = np.array([(1,2), (2,1)]) # Prior to generate lambdas (a,b)

# Init the model
model = FetalChain(alpha, theta)

# Set the exact priors of the example
lambdas = np.array([0.256, 3.101])
P = np.array([(0.984, 0.016), (0.308, 0.692)])
model.set_priors(P, lambdas)

# Running MCMC (take some time)
F, S = MCMC(model, Y, N = 6000, N_rejected = 200)



# Computing stats
history = model.get_history()
stats = {}
stats["Mean"] = np.mean(history, axis = 0).round(3).flatten()
stats["Std"] = np.std(history, axis = 0).round(3).flatten()
stats["Min"] = np.min(history, axis = 0).round(3).flatten()
stats["Max"] = np.max(history, axis = 0).round(3).flatten()
stats = pd.DataFrame.from_dict(stats, orient="columns")
print(stats)

# Plotting final probabilities
n = len(F)    
plt.figure()
plt.subplot(2,1,1)
plt.bar(range(n), F[:, 0])
plt.subplot(2,1,2)
plt.bar(range(n), F[:, 1])


