import numpy as np

def solver_ns2d(N=64, steps=300, dt=0.001, nu=0.01, I_M=0.0025, amplitude=3):
    """2D incompressible spectral solver with curvometric force."""
    # Wave numbers
    k = np.fft.fftfreq(N, 1.0 / N) * 2 * np.pi
    kx, ky = np.meshgrid(k, k)
    k2 = kx**2 + ky**2
    k2[0, 0] = 1  # regularize zero mode

    # Curvature field
    x = np.linspace(0, 2*np.pi, N)
    y = np.linspace(0, 2*np.pi, N)
    X, Y = np.meshgrid(x, y)
    raw_field = amplitude * np.sin(8*X) * np.sin(4*Y)
    kappa = I_M + np.clip(raw_field, -1, 1)

    # Initial divergence-free velocity
    u = np.random.randn(N, N)
    v = np.random.randn(N, N)
    u_hat = np.fft.fft2(u)
    v_hat = np.fft.fft2(v)
    div = kx * u_hat + ky * v_hat
    u_hat -= kx * div / k2
    v_hat -= ky * div / k2

    # Time stepping
    for _ in range(steps):
        u = np.fft.ifft2(u_hat).real
        v = np.fft.ifft2(v_hat).real

        grad_kx, grad_ky = np.gradient(kappa)
        Fx, Fy = -grad_kx, -grad_ky

        # Nonlinear terms
        ux = np.fft.ifft2(1j * kx * u_hat).real
        uy = np.fft.ifft2(1j * ky * u_hat).real
        vx = np.fft.ifft2(1j * kx * v_hat).real
        vy = np.fft.ifft2(1j * ky * v_hat).real
        nonlin_u = np.fft.fft2(u * ux + v * uy)
        nonlin_v = np.fft.fft2(u * vx + v * vy)

        # Viscous terms
        visc_u = -nu * k2 * u_hat
        visc_v = -nu * k2 * v_hat

        # Curvometric force in spectral space
        force_u = np.fft.fft2(Fx)
        force_v = np.fft.fft2(Fy)

        # Right-hand sides
        rhs_u = -nonlin_u + visc_u + force_u
        rhs_v = -nonlin_v + visc_v + force_v

        # Enforce incompressibility
        div_rhs = kx * rhs_u + ky * rhs_v
        rhs_u -= kx * div_rhs / k2
        rhs_v -= ky * div_rhs / k2

        # Advance
        u_hat += dt * rhs_u
        v_hat += dt * rhs_v

    # Compute final fields and diagnostics
    u_final = np.fft.ifft2(u_hat).real
    v_final = np.fft.ifft2(v_hat).real
    energy = 0.5 * np.mean(u_final**2 + v_final**2)
    vel_mean = np.mean(np.sqrt(u_final**2 + v_final**2))
    return energy, vel_mean
