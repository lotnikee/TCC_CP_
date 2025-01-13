# Calculate the radial distribution function (RDF) for 1s, 2s and 2p orbitals,
# and then normalise the result so the maximum value of each RDF is one. 
# The RDF is 4*π*r^2 times the square of the wavefunction. 
# The (unnormalised) wavefunctions themselves are given by: 
    
    # phi_1s = e^(-r/2)
    # phi_2s = (2-r)e^(-r/2)
    # phi_2p = (6-6r+r^2)e^(-r/2)
    
    
# Import the different libraries used in this project
import math
import numpy as np
import matplotlib.pyplot as plt

# Define a range for the radius
r = np.arange(0, 20.0, 0.05)

# Define the RDF for 1s, 2s and 2p orbitals
def RDF_1s(r):
    phi_1s = np.exp(-r/2)
    return (4 * math.pi * (r**2)) * (phi_1s ** 2)

def RDF_2s(r):
    phi_2s = (2-r) * np.exp(-r/2)
    return (4 * math.pi * (r**2)) * (phi_2s ** 2)

def RDF_2p(r):
    phi_2p = (6 - (6*r) + (r**2)) * np.exp(-r/2)
    return  (4 * math.pi * (r**2)) * (phi_2p ** 2)

# Normalise the RDF so the maximum peak value is 1
max_RDF_1s = np.max(RDF_1s(r))
RDF_normalised_1s = RDF_1s(r) / max_RDF_1s

max_RDF_2s = np.max(RDF_2s(r))
RDF_normalised_2s = RDF_2s(r) / max_RDF_2s

max_RDF_2p = np.max(RDF_2p(r))
RDF_normalised_2p = RDF_2p(r) / max_RDF_2p


# Plot the normalised RDF for the 1s, 2s and 2p orbitals
plt.figure(figsize=(15, 12))
plt.xlabel("Radius in Å")
plt.ylabel("Radial Distribution Function")
plt.title("Normalised Radial Distribution Functions")
plt.plot(r, RDF_normalised_1s, label="1s", color="blue", linestyle="--", linewidth=2)
plt.plot(r, RDF_normalised_2s, label="2s", color="green", linestyle="--", linewidth=2)
plt.plot(r, RDF_normalised_2p, label="2p", color="red", linestyle="--", linewidth=2)
plt.legend(loc="upper right")
plt.grid(True)
plt.show()


# Define the RDF for the 3p and 3d orbitals 
def RDF_3p(r): 
    phi_3p = ((4 * r) - (r**2)) * np.exp(-r/2)
    return (4 * math.pi * (r**2)) * (phi_3p ** 2)

def RDF_3d(r):
    phi_3d = (r**2) * np.exp(-r/2)
    return (4 * math.pi * (r**2)) * (phi_3d ** 2)

# Normalise the RDF for the 3p and 3d orbitals 
max_RDF_3p = np.max(RDF_3p(r))
RDF_normalised_3p = RDF_3p(r) / max_RDF_3p

max_RDF_3d = np.max(RDF_3d(r))
RDF_normalised_3d = RDF_3d(r) / max_RDF_3d

# Plot the normalised RDF for the 3p and 3d orbitals
plt.figure(figsize=(15,12))
plt.xlabel("Radius in Å")
plt.ylabel("Radial Distribution Function")
plt.title("Normalised Radial Distribution Functions")
plt.plot(r, RDF_normalised_3p, label="3p", color="purple", linestyle="--", linewidth=2)
plt.plot(r, RDF_normalised_3d, label="3d", color="pink", linestyle="--", linewidth=2)
plt.legend(loc="upper right")
plt.grid(True)
plt.show()


# Set variables for minimum, maximum and number of intervals 
a = 0 
b = 20
n = 1000

# Define functions to integrate orbitals using Simpson's rule 
def simpsons_rule(func, a, b, n):
    
    if n % 2 != 0:
        raise ValueError ("Number of intervals (n) must be even for Simpson's rule.")
    
    h = (b-a) / n
    x = np.linspace(a, b, n + 1)
    y = func(x)
    
    integral = y[0] + y[-1]
    integral += 4*sum(y[1:-1:2])
    integral += 2*sum(y[2:-2:2])
    integral *= h/3

    return integral 

result_1s = simpsons_rule(RDF_1s, a, b, n)
result_2s = simpsons_rule(RDF_2s, a, b, n)
result_2p = simpsons_rule(RDF_2p, a, b, n)
print("Approximate integral of 1s orbital is:", result_1s)
print("Approximate integral of 2s orbital is:", result_2s)
print("Approximate integral of 2p orbital is:", result_2p, "\n")

# Define function to integrate orbitals using Trapezoidal Rule
def trapezoidal_rule(func, a, b, n):
    h = (b-a)/n
    x = np.linspace(a,b,n+1)
    y = func(x)
    return h*(np.sum(y) - 0.5*(y[0]+y[-1]))

result_trap_1s = trapezoidal_rule(RDF_1s, a, b, n)
result_trap_2s = trapezoidal_rule(RDF_2s, a, b, n)
result_trap_2p = trapezoidal_rule(RDF_2p, a, b, n)
print("Integral of 1s orbital using trapezoidal rule is:", result_trap_1s)
print("Integral of 2s orbital using trapezoidal rule is:", result_trap_2s)
print("Integral of 2p orbital using trapezoidal rule is", result_trap_2p, "\n")

# Compute the normalisation constants for the different orbitals
normal_c_1s = 1 / np.sqrt(result_trap_1s)
normal_c_2s = 1 / np.sqrt(result_trap_2s)
normal_c_2p = 1 / np.sqrt(result_trap_2p)
print("Normalisation constant for a 1s orbital is:", normal_c_1s)
print("Normalisation constant for a 2s orbital is:", normal_c_2s)
print("Normalisation constant for a 2p orbital is:", normal_c_2p)


