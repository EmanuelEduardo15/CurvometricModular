import numpy as np

def sample_sphere(n_samples, dim=8):
    """Uniform sampling on S^{dim-1}."""
    vec = np.random.normal(size=(n_samples, dim))
    vec /= np.linalg.norm(vec, axis=1)[:, None]
    return vec

def modular_curvature(x):
    """Example modular curvature."""
    return np.sin(10 * x[0]) * np.cos(5 * x[1]) + 0.02 * x[2]**2

def compute_IM(n_samples):
    """Compute curvometric invariant via Monte Carlo."""
    samples = sample_sphere(n_samples)
    values = np.array([modular_curvature(x) for x in samples])
    return float(np.mean(values))
