import math
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# landmarks
t0 = np.array([12,4]) #tower 0
t1 = np.array([5,7]) #tower 1
m0 = np.array([10,8]) #uni
m1 = np.array([6,3]) #home

def likelihood(m):
    """
    Calculates the likelihood of my friend is at place m
    """
    d0 = 3.9 # measurement 0
    d1 = 4.5 # measurement 1
    var0 = 1
    var1 = 1.5

    # Find distance (d_hat) between location 'm' with tower 't'
    d0_hat = math.sqrt(np.sum(np.square(m-t0)))
    d1_hat = math.sqrt(np.sum(np.square(m-t1)))

    # Evaluate sensor model a.k.a pdf
    pdf0 = scipy.stats.norm.pdf(d0, d0_hat, math.sqrt(var0))
    pdf1 = scipy.stats.norm.pdf(d1, d1_hat, math.sqrt(var1))

    return pdf0*pdf1

# create test points around POI
x = np.arange(3.0, 15.0, 0.5)
y = np.arange(-5.0, 15.0, 0.5)
X,Y = np.meshgrid(x,y)

# calculate likelihood for each of the points
z = np.array([likelihood(np.array([x,y])) for x,y in zip(X.flatten(), Y.flatten())])
Z = z.reshape(X.shape)
#print(f'X: {X.shape}, Y: {Y.shape}, Z: {Z.shape}')
#X: (40, 24), Y: (40, 24), Z: (40, 24)

# plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X,Y,Z, rstride=1, cstride=1, cmap=cm.coolwarm, alpha=0.5)
## Additional points
ax.scatter(m0[0], m0[1], likelihood(m0), c='g', marker='o', s=100) #uni
ax.scatter(m1[0], m1[1], likelihood(m1), c='r', marker='o', s=100) #home
## towers
ax.scatter(t0[0], t0[1], likelihood(t0), c='g', marker='^', s=100)
ax.scatter(t1[0], t1[1], likelihood(t1), c='g', marker='^', s=100)

ax.set_xlabel('mX')
ax.set_ylabel('mY')
ax.set_zlabel('likelihood')

plt.show()
