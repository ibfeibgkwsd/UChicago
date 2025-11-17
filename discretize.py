import numpy as np
import matplotlib.pyplot as plt

K_min = 0.001
K_max = 3
beta = 0.9
A_vals = [0.8, 1.2]                                             # A(z)

V = np.zeros((500, 2))                                          # list with lists [0, 0], V[i, j] = V(K_i, z_j)
Kp_policy = np.zeros((500, 2))                                  # list with lists [0, 0], Kp[i, j] = optimal allocation K_i, z=1, K_i, z=2
K_grid = np.linspace(K_min, K_max, 500)                         # list with 500 equally spaced numbers

def f(K, z_idx):                                                # production function
    A = A_vals[z_idx]                            
    return (A * K ** 0.3) + (0.3 * K)

def interpolate(Kp, V, K_grid):                                # linear interpolation of V(K', z) for z = 1, 2 using np.interp
    v1 = np.interp(Kp, K_grid, V[:, 0])
    v2 = np.interp(Kp, K_grid, V[:, 1])
    return v1, v2


for k in range(500):
    V_old = V.copy()
    V_new = np.empty_like(V)
    Kp_new = np.empty_like(Kp_policy)

    for i in range(500):                                        # iterate over K_grid                                        
        K = K_grid[i]                                           
        for z in range(2):
            output = f(K, z)
            Kp_max = min(output - 1e-8, K_max)                  # use arbitrarily small values to avoid log(0)
            Kp_candidates = np.linspace(K_min, Kp_max, 100)     # uniformly select 100 potential candidates for K', bounded by 0 <= K' <= f(K, z) - C
            best_val = -1e12
            best_Kp = K_min
            for Kp in Kp_candidates:                            # iterate over candidates, interpolate V(K')
                C = output - Kp
                u = np.log(C)

                V1, V2 = interpolate(Kp, V_old, K_grid)
                EV = 0.5 * V1 + 0.5 * V2

                val = u + beta * EV

                if val > best_val:
                    best_val = val
                    best_Kp = Kp
        
            V_new[i, z] = best_val
            Kp_new[i, z] = best_Kp
    
    V = V_new
    Kp_policy = Kp_new

plt.plot(K_grid, Kp_policy[:, 0], label ="K'(K, z=1)")
plt.plot(K_grid, Kp_policy[:, 1], label ="K'(K, z=2)")
plt.legend
plt.show()

C_policy = np.zeros_like(Kp_policy)                             # consumption policy is determined by f(K, z) = C + K'
for i in range(500):
    K = K_grid[i]
    for z in range(2):
        output = f(K, z)
        Kp = Kp_policy[i, z]
        C_policy[i, z] = output - Kp

plt.plot(K_grid, C_policy[:, 0], label ="K'(K, z=1)")
plt.plot(K_grid, C_policy[:, 1], label ="K'(K, z=2)")
plt.legend
plt.show()