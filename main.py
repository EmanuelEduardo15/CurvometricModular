from modules.curvometry import compute_IM
from modules.solver import solver_ns2d

def main():
    # Exemplo de execução
    I_M = compute_IM(100000)
    energy, vel = solver_ns2d(N=64, steps=300, I_M=I_M, amplitude=3)
    print(f"Invariante curvométrico I_M ≈ {I_M:.6f}")
    print(f"Energy ≈ {energy:.6e}, VelMean ≈ {vel:.6f}")

if __name__ == "__main__":
    main()
