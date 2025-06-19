# CurvometricModular

Protótipo de Curvometria Modular Aplicada à Dinâmica de Fluidos.

## Descrição

Este repositório contém um modelo matemático e numérico para:

- Estimar um invariante curvométrico \(I_M\) via Monte Carlo em \(S^7\).
- Simular a dinâmica de fluidos 2D (Navier–Stokes) com força curvométrica.
- Interface simples via CLI em `main.py`.

## Estrutura

- `main.py`: script principal de demonstração.
- `modules/curvometry.py`: funções de amostragem e cálculo de \(I_M\).
- `modules/solver.py`: solver espectral 2D modificado.
- `requirements.txt`: dependências do Python.

## Instalação

```bash
git clone https://github.com/EmanuelEduardo15/CurvometricModular.git
cd CurvometricModular
pip install -r requirements.txt
