import math
import numpy as np
import matplotlib.pyplot as plt

# Define a radius range and a radial factor 
r = np.arange(0, 30.0, 0.05)
radial_factor = 4 * math.pi * (r ** 2)

# Define the wavefunctions for the different orbitals in a dictionary
orbitals = {
    "1s": lambda r: np.exp(-r),
    "2s": lambda r: (2 - r) * np.exp(-r/2),
    "2p": lambda r: r * np.exp(-r/2)}

# Function to compute the RDFs
def compute_rdf(wavefunc, r, radial_factor):
    return radial_factor * (wavefunc(r) ** 2)

# Function to normalise the RDFs so that its maximum value is 1
def normalise_max(rdf):
    return rdf / np.max(rdf)

# Store the results of the normalised RDFs in a dictionary
rdf_normalised_max = {}

# Run a for loop to compute the RDFs and their normalisation 
for name, wavefunc in orbitals.items():
    rdf = compute_rdf(wavefunc, r, radial_factor)
    rdf_normalised_max[name] = normalise_max(rdf)
        
# Append the orbitals dictionary to include the additional 3p and 3d orbital 
additional_orbitals = {
    "3p": lambda r: ((6 * r) - (r ** 2)) * np.exp(-r/3),
    "3d": lambda r: (r ** 2) * np.exp(-r/3)} 
orbitals.update(additional_orbitals)

# Run the for loop again to compute all the RDFs (including 3p and 3d) and their normalisation
for name, wavefunc in orbitals.items():
    rdf = compute_rdf(wavefunc, r, radial_factor)
    rdf_normalised_max[name] = normalise_max(rdf)
    
# Function to perform Simpson's rule integration
def simpsons_rule(func, a, b, n):
    if n % 2 != 0:
        raise ValueError("Number of intervals (n) must be even for Simpson's rule.")
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = func(x)
    integral = y[0] + y[-1] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-2:2])
    integral *= h / 3
    return integral

# Function to perform Trapezium rule integration
def trapezium_rule(func, a, b, n):
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = func(x)
    integral = h * (np.sum(y) - 0.5 * (y[0] + y[-1]))
    return integral

# Integration parameters
a = 0
b = 30
n = 1000

# Dictionaries to store results
rdf_normalised_integral = {}
normalisation_constants = {}
rdf_normalised_integral_norm = {}

for name, wavefunc in orbitals.items():
    rdf = compute_rdf(wavefunc, r, radial_factor)
    rdf_normalised_max[name] = normalise_max(rdf)
    
    # Define a function for integration (RDF)
    def rdf_func(x, wf=wavefunc):
        return compute_rdf(wf, x, 4 * math.pi * (x ** 2))
    
    # Integrate using Simpson's rule
    integral_simpson = simpsons_rule(rdf_func, a, b, n)
    
    # Integrate using Trapezium rule
    integral_trap = trapezium_rule(rdf_func, a, b, n)
    
    # Compute normalisation constant using Trapezium integral
    normalisation_constants[name] = 1 / integral_trap
    
    # Store normalised RDF where integral equals 1
    rdf_normalised_integral[name] = normalisation_constants[name] * rdf

# Print Integration Results
print("Integration Results using Simpson's Rule:")
for name, wavefunc in orbitals.items():
    integral_simpson = simpsons_rule(
        lambda x: compute_rdf(wavefunc, x, 4 * math.pi * (x ** 2)),
        a, b, n)
    print(f"Integral of {name} orbital: {integral_simpson:.5f}")

print("\nIntegration Results using Trapezium Rule:")
for name, wavefunc in orbitals.items():
    integral_trap = trapezium_rule(
        lambda x: compute_rdf(wavefunc, x, 4 * math.pi * (x ** 2)),
        a, b, n)
    print(f"Integral of {name} orbital: {integral_trap:.5f}")

# Print normalisation constants
print("\nNormalisation Constants (1 / Trapezium Integral):")
for name, constant in normalisation_constants.items():
    print(f"Normalisation constant for {name} orbital: {constant:.5f}")

# Create subplots: 1 row, 2 columns
fig, axs = plt.subplots(1, 2, figsize=(18, 8))

# First subplot: Max Normalised RDFs
for name, rdf in rdf_normalised_max.items():
    axs[0].plot(r, rdf, label=name, linewidth=2)
axs[0].set_xlabel("Radius in Å")
axs[0].set_ylabel("Radial Distribution Function")
axs[0].set_title("RDFs Normalised by Maximum Value")
axs[0].legend()
axs[0].grid(True)

# Second subplot: Integral Normalised RDFs
for name, rdf in rdf_normalised_integral.items():
    axs[1].plot(r, rdf, label=name, linewidth=2)
axs[1].set_xlabel("Radius in Å")
axs[1].set_ylabel("Radial Distribution Function")
axs[1].set_title("RDFs Normalised by Integral")
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()
plt.show()
