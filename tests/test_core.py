import math
import numpy as np
import pytest

from modules.curvometry import compute_IM
from modules.solver import solver_ns2d

@pytest.mark.parametrize("n_samples", [1000, 5000])
def test_compute_IM_returns_reasonable_value(n_samples):
    im = compute_IM(n_samples)
    # I_M deve ser um float finito e dentro de um intervalo plausível [-1,1]
    assert isinstance(im, float)
    assert math.isfinite(im)
    assert -1.0 <= im <= 1.0

@pytest.mark.parametrize("params", [
    {"N": 32,  "steps": 50,  "I_M": 0.0, "amplitude": 1},
    {"N": 64,  "steps": 100, "I_M": 0.1, "amplitude": 3},
])
def test_solver_ns2d_returns_finite_metrics(params):
    energy, vel = solver_ns2d(
        N=params["N"],
        steps=params["steps"],
        dt=0.001,
        nu=0.01,
        I_M=params["I_M"],
        amplitude=params["amplitude"]
    )
    # Ambos devem ser floats finitos e não-negativos
    assert isinstance(energy, float) and math.isfinite(energy) and energy >= 0.0
    assert isinstance(vel, float)   and math.isfinite(vel)   and vel >= 0.0

def test_end_to_end_small_run(tmp_path, capsys):
    # Simula executar o main.py com um grid pequeno
    from modules.curvometry import compute_IM
    from modules.solver import solver_ns2d

    im = compute_IM(1000)
    energy, vel = solver_ns2d(N=16, steps=10, I_M=im, amplitude=1)
    # Captura os valores
    assert math.isfinite(im) and math.isfinite(energy) and math.isfinite(vel)
    # Exibe para inspeção manual
    print(f"I_M={im:.6f}, E={energy:.6e}, V={vel:.6f}")
    captured = capsys.readouterr()
    assert "I_M=" in captured.out
