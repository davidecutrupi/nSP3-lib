# nSP3-lib

## Introduction

`nSP3-lib` is a matrix-free finite element solver for multigroup simplified P3
neutron eigenvalue problems. It supports continuous and discontinuous finite
element discretizations and targets large-scale reactor-core calculations with
geometric multigrid preconditioning and hp-refinement on a single mesh.

Benchmark geometry, material data, and cross sections are read from JSON files.
The main numerical options are controlled through `params.prm`, including the
benchmark, FE family, refinement strategy, eigensolver tolerance, group solver
tolerance, and multigrid coarse-grid options.

A more detailed report is available in the repository.


## Mathematical Formulation

For each energy group `g`, the symmetric SP3 formulation is written in terms of
two unknowns:

```text
u0_g = phi0_g + 2 phi2_g
u2_g = 3 phi2_g
```

or, equivalently,

```text
phi0_g = u0_g - (2/3) u2_g
phi2_g = (1/3) u2_g
```

With `u_g = [u0_g, u2_g]^T`, the multigroup eigenvalue problem can be written
group by group as:

```text
- div(D_g grad u_g) + R_g u_g =
  sum_{g' != g} S_{g' -> g} u_g'
  + (chi_g / k_eff) sum_{g'=1..G} F_g' u_g'
  + q_ext_g
```

The diffusion and removal matrices are:

```text
D_g =
[ D0_g        0          ]
[ 0     (5/9) D2_g       ]

R_g =
[ sigma_R0_g                  -(2/3) sigma_R0_g                  ]
[ -(2/3) sigma_R0_g   (4/9) sigma_R0_g + (5/9) sigma_R2_g       ]
```

The scattering and fission source matrices use the same SP3 mode-coupling
matrix:

```text
S_{g' -> g} =
  sigma_s0_{g' -> g}
  [  1    -2/3 ]
  [ -2/3   4/9 ]

F_g =
  nu_sigma_f_g
  [  1    -2/3 ]
  [ -2/3   4/9 ]
```

Here `D0_g` and `D2_g` are the SP3 diffusion coefficients, `sigma_R0_g` and
`sigma_R2_g` are removal cross sections for the zeroth and second moments,
`sigma_s0_{g' -> g}` is the zeroth-moment scattering cross section from group
`g'` to group `g`, `nu_sigma_f_g` is the fission production cross section,
`chi_g` is the fission spectrum, and `k_eff` is the multiplication factor. For
homogeneous eigenvalue calculations, the external source `q_ext_g` is zero.

In compact operator notation, the left-hand side is the group operator
`A_g u_g`, with:

```text
A_g u_g = - div(D_g grad u_g) + R_g u_g
```

The DG discretization applies an interior-penalty weak form to this operator.
It includes cell diffusion and removal terms, internal-face consistency terms,
penalty terms, and Robin-type SP3 boundary contributions. The code also
supports CG elements; in the repository this is selected by `FE type = CG`,
while `FE type = DG` uses discontinuous elements and interior-face terms.

The outer eigenvalue iteration is a power iteration. Each iteration builds the
group right-hand sides from the latest scattering and fission sources, solves
the group systems, updates `k_eff` from the total fission source, and
normalizes the stored old solutions. For goal-oriented refinement, the adjoint
problem is solved by reversing the energy sweep and transposed couplings.

## Repository Structure

The repository has the following main pieces:

- `CMakeLists.txt`: builds the `neutronics` executable, requires C++17 and
  `deal.II` 9.7.1, and fetches `nlohmann/json` v3.11.3 through CMake
  `FetchContent`.
- `params.prm`: runtime configuration for the benchmark, output path, FE type,
  refinement cycles, p-refinement limits, eigensolver tolerance, group solver
  tolerance, and multigrid coarse-solver options.
- `src/main.cpp`: initializes MPI, constructs `SolverParameters`, selects the
  benchmark JSON file from `../benchmarks/<benchmark>.json`, reads its
  dimension, and dispatches to `NeutronSolver<1>`, `NeutronSolver<2>`, or
  `NeutronSolver<3>`.
- `include/SolverParameters.hpp`: declares, parses, and validates entries from
  `../params.prm`.
- `include/NeutronSolver.hpp` and `src/NeutronSolver.cpp`: own mesh creation,
  energy-group setup, the forward and adjoint eigenvalue loops, h/p refinement,
  result output, and optional power-distribution CSV output.
- `include/EnergyGroup.hpp` and `src/EnergyGroup.cpp`: manage one energy group:
  DoF setup, matrix-free system setup, RHS construction, group solves,
  h-transfer and p-transfer, fission-source integration, and VTU/PVTU output.
- `include/MaterialData.hpp`, `include/GeometryData.hpp`, and
  `include/CrossSectionManager.hpp`: parse JSON material and geometry data,
  expose cross sections and boundary data, and cache material coefficients in
  matrix-free-friendly vectorized arrays.
- `include/ZeroModeOperator.hpp`, `include/SecondModeOperator.hpp`,
  `include/CouplingOperator.hpp`, and `include/SP3Operator.hpp`: matrix-free
  operators for scalar SP3 blocks, mode coupling, and the monolithic two-block
  SP3 system.
- `include/BlockGSPreconditioner.hpp`,
  `include/BlockDiagonalPreconditioner.hpp`,
  `include/MultigridPreconditioner.hpp`, and
  `include/MultigridWrappers.hpp`: preconditioning, smoothers, transfer
  wrappers, coarse-grid assembly, and Trilinos direct coarse-grid wrappers.
- `benchmarks/`: JSON input cases, including `slab1d_het.json`, `biblis.json`,
  `lwr.json`, `pwr.json`, and `data.json`.
- `deal.II_AMR/`: modified `deal.II` matrix-free files used for DG adaptive
  mesh refinement support.

## Build and Run

Requirements:

- CMake 3.20 or newer.
- A C++17 compiler.
- `deal.II` 9.7.1.

Example build:

```sh
git clone https://github.com/davidecutrupi/nSP3-lib.git
cd nSP3-lib
mkdir build out
cd build
cmake ..
make -j
```

If CMake cannot find `deal.II`, pass its installation path explicitly with
`-DDEAL_II_DIR=/path/to/dealii`.

The `out/` directory must exist before running. The solver writes into
`Output directory` from `params.prm`, but it does not create that directory.
The executable also expects to be run from `build/` with the default layout,
because the source reads `../params.prm` and
`../benchmarks/<benchmark>.json`.

Run with MPI:

```sh
mpirun -np 4 ./neutronics
```

To run simulations with DG and adaptive mesh refinement, the files inside
`deal.II_AMR/` must replace the corresponding files in the `deal.II` 9.7.1
source tree. After replacing them, the `deal.II` library must be compiled
again.

Edit `params.prm` to select:

- `Benchmark`: one of the JSON benchmark names, such as `slab1d_het`,
  `biblis`, `lwr`, or `pwr`.
- `FE type`: `DG` or `CG`.
- `Spatial Refinement Type`: `global`, `adaptive`, `goal`, or `none`.
- p-refinement controls: maximum polynomial degree, thermal-group count,
  thermal maximum polynomial degree, and p-refinement threshold fraction.
- outer eigenvalue iteration and per-group solver tolerances.
- multigrid p-coarsening and direct coarse-solver options.

## Triangulation and Mesh Refinement Strategies

The SP3 equations are naturally block structured. `nSP3-lib` stores the two
SP3 modes in block vectors instead of interleaving both components in a single
large vector. This layout preserves an important invariant: the zeroth and
second modes are defined on the same scalar degree-of-freedom space inside
each energy group. Therefore, every energy group owns one scalar DoF handler,
and all scalar and block operators are built around it.

For `N` scalar degrees of freedom, the group system has size `2N x 2N`:

```text
A =
[ A00  A02 ]
[ A20  A22 ]
```

Each block is an `N x N` operator. The block `A00` contains the zeroth-mode
diffusion and removal terms, while `A22` contains the second-mode correction
terms. The off-diagonal blocks `A02` and `A20` represent the local coupling
between the two SP3 modes. Custom wrappers are used wherever a block vector
must interact with scalar mode operators, smoothers, transfers, or coarse-grid
objects.

Mesh handling is central to the performance of the solver. Multigroup
neutronics produces fluxes with different spatial behavior: fast fluxes are
typically smoother, while thermal fluxes can contain sharper gradients,
especially near material interfaces, reflectors, and absorbers. A mesh that is
appropriate for the fast group may be too coarse for the thermal group, while
refining the whole domain too aggressively increases cost without necessarily
improving the relevant quantity of interest.

The solver adopts a single-mesh hp strategy. This avoids the interpolation and
cached prolongation machinery that would be required if each energy group were
solved on its own mesh. With a single geometry, scattering and fission terms
remain integrals between functions defined on the same cells, even when the
groups use different polynomial degrees. Spatial refinement is then used to
capture discontinuities and current gradients, while group-wise p-refinement
adjusts the polynomial degree for each energy group.

The adaptive h-refinement indicator is based on a Kelly estimator. In scalar
form, for group `g` and cell `K`, the estimator is:

```text
rho_{g,K} =
  sqrt(h_K) || jump(D_g n . grad(phi_{g,h})) ||_{boundary of K}
```

Here `h_K` is the cell diameter, `D_g` is the diffusion coefficient, and the
jump measures the discontinuity of the normal current across cell faces. The
implementation refines the fraction `0.3` of cells with the largest estimated
errors and coarsens the fraction `0.03` with the smallest estimated errors
when `Spatial Refinement Type = adaptive`.

The p-refinement decision is made at energy-group level. Let `eta_g` be the
global error indicator for group `g`, `G` the total number of groups, and
`alpha_p` the user-selected base threshold. The degree `p_g` is increased when:

```text
eta_g > alpha_{p,g} max_k eta_k
and
p_g < p_max,g
```

with the group-dependent threshold:

```text
alpha_{p,g} = alpha_p * g / (G + 1)
```

using one-based group numbering. This makes the p-refinement decision depend
on the group position and allows thermal groups to be penalized through lower
maximum polynomial degrees. In the input file this is controlled by
`Max polynomial degree`, `Thermal group count`, `Thermal max polynomial degree`,
and `P refinement threshold fraction`.

A goal-oriented refinement option is also available. In this mode the solver
computes an adjoint eigenproblem and uses `k_eff` as the quantity of interest.
The adjoint problem is obtained by transposing the energy couplings. In scalar
diffusion notation, the adjoint group equation can be read as:

```text
- div(D_g grad(phi_g^dagger)) + sigma_R,g phi_g^dagger =
  sum_{g' != g} sigma_s0_{g -> g'} phi_g'^dagger
  + (nu_sigma_f,g / k_eff) sum_{g'=1..G} chi_g' phi_g'^dagger
```

The removal cross section still contains absorption and out-scattering from
the current group. The scattering term uses the forward transition
`g -> g'`, because the energy-coupling matrix is transposed in the adjoint
problem. The fission term is transposed in the same way: the production cross
section belongs to the current group, while the adjoint fluxes of all groups
are weighted by their fission spectra.

Once the primal and adjoint solutions are available, the local goal-oriented
indicator is the product of the two unweighted Kelly indicators:

```text
rho_goal_{g,K} = rho_primal_{g,K} * rho_dual_{g,K}
```

In expanded form this corresponds to multiplying the face-jump estimator built
from the forward flux by the same estimator built from the adjoint flux. This
focuses refinement on cells that affect the selected quantity of interest, not
only on cells where the primal solution has a large local jump.

## Preconditioning Strategies

The SP3 group systems are solved with preconditioned Krylov methods. The key
ingredient is geometric multigrid, which only requires matrix-vector products
on each level and is therefore compatible with the matrix-free implementation.
Two SP3 preconditioning paths are implemented: a block Gauss-Seidel
preconditioner built from scalar mode solvers, and a monolithic multigrid
preconditioner for the full two-block SP3 system.

For the block strategy, the SP3 matrix is viewed as:

```text
A =
[ A00  A02 ]
[ A20  A22 ]
```

The block Gauss-Seidel preconditioner uses the lower triangular part:

```text
P_GS =
[ A00   0  ]
[ A20  A22]

P_GS^{-1} =
[ A00^{-1}                         0        ]
[ -A22^{-1} A20 A00^{-1}           A22^{-1} ]
```

For transposed application, the analogous upper-triangular coupling is used.
This preconditioner accounts for one off-diagonal mode-coupling block while
keeping the subsolves scalar. Each subsystem can be solved with CG or a
Chebyshev-type iteration because the individual mode operators are symmetric.
Those scalar subsolves are themselves preconditioned with geometric multigrid.
In the code this path is represented by `BlockGSPreconditioner`,
`ZeroModeOperator`, `SecondModeOperator`, `CouplingOperator`, and
`MultigridPreconditioner`.

The alternative path constructs and solves the monolithic SP3 system. The
matrix-free operator is `SP3Operator`, and the preconditioner is a geometric
multigrid method acting on two-block vectors. Custom wrappers apply the
smoother, restriction, and prolongation operations consistently on both SP3
blocks.

At the coarsest level, the monolithic matrix is assembled by merging the four
coarse block matrices:

```text
A_c =
[ A00,c  A02,c ]
[ A20,c  A22,c ]
```

The coarse sparsity pattern accounts for the fact that the off-diagonal blocks
do not contain interior-face contributions. The implementation builds this
matrix efficiently by using direct access to the coarse matrices.

The monolithic smoother uses a block-Jacobi preconditioner that includes the
local interaction between the zeroth and second modes. For each scalar DoF
`i`, the inverse of the local `2 x 2` diagonal block is:

```text
P_BJ,i^{-1} =
1 / (d00_i d22_i - d02_i d20_i)
*
[  d22_i  -d02_i ]
[ -d20_i   d00_i ]
```

Here `d00_i`, `d02_i`, `d20_i`, and `d22_i` are diagonal entries extracted
from the corresponding SP3 blocks. This is more informative than applying two
independent scalar Jacobi preconditioners, because it keeps the local coupling
between modes inside the smoother.

Both preconditioning paths use a global-coarsening multigrid hierarchy. The
hierarchy starts from the active mesh and moves to coarser globally defined
meshes. This improves the homogeneity of work across MPI processes compared
with local coarsening and avoids refinement-edge matrices at multigrid levels,
because each level is a complete mesh over the whole domain.

Since the solver also supports p-refinement, the coarse matrix can still be
large if the active polynomial degree is high. For this reason the multigrid
hierarchy can add global p-coarsened levels when the coarsest h-level has a
polynomial degree greater than or equal to `Coarse p-coarsening min degree`
and the number of DoFs is larger than `Coarse p-coarsening min dofs`.
Available polynomial coarsening sequences are:

- `bisect`: halves the polynomial degree at each level;
- `decrease_by_one`: reduces the polynomial degree by one at each level;
- `go_to_one`: jumps directly to degree one.

The coarsest level is solved by a direct solver. For small systems the code
uses Amesos KLU, which has strong serial performance but limited scalability.
When the matrix size is greater than or equal to `Coarse direct KLU max dofs`,
the coarse solve switches to Amesos MUMPS. The parameters `MUMPS ICNTL(14)`,
`MUMPS ICNTL(4)`, and `MUMPS out of core` are exposed in `params.prm` to manage
factorization memory, output verbosity, and optional out-of-core behavior.

## Numerical Results

### 1D Heterogeneous Slab Geometry

The first benchmark is a seven-region heterogeneous slab with total length
18 cm and vacuum boundary conditions at the two external faces. The geometry
is a repeated fuel-reflector arrangement. Fuel regions are 2.4 cm thick, while
reflector regions are 2.7 cm thick.

Seven-region slab cross sections:

| Material | nu sigma_f (cm^-1) | sigma_s (cm^-1) | sigma_t (cm^-1) |
| --- | ---: | ---: | ---: |
| Fuel | 0.178 | 0.334 | 0.416667 |
| Reflector | 0.0 | 0.334 | 0.370370 |

The computed flux and current profiles are coherent with the reference
calculation. The eigenvalue obtained by `nSP3-lib` is `k_eff = 1.148744`,
which matches the P3 nodal result to the shown precision and improves
significantly over the P1 nodal approximation.

| Method | k_eff |
| --- | ---: |
| P1 nodal | 1.11387 |
| P3 nodal | 1.14874 |
| P5 nodal | 1.15736 |
| nSP3-lib | 1.14874 |
| Reference | 1.16224 |

### BIBLIS 2D Benchmark

The BIBLIS benchmark is a realistic two-group problem representative of an
operating pressurized water reactor. It contains 257 homogenized fuel
assemblies with width 23.1226 cm. Eight materials are used in total, including
the reflector, and the external boundary condition is vacuum. The realistic
core layout makes this case useful for testing coarse-mesh and adaptive
strategies.

Macroscopic cross sections for BIBLIS:

| Material | Group | D | sigma_a | nu sigma_f | sigma_f | sigma_12 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 1.4360 | 0.0095042 | 0.0058708 | 0.0023768 | 0.017754 |
| 1 | 2 | 0.3635 | 0.0750580 | 0.0960670 | 0.0388940 | - |
| 2 | 1 | 1.4366 | 0.0096785 | 0.0061908 | 0.0025064 | 0.017621 |
| 2 | 2 | 0.3636 | 0.0784360 | 0.1035800 | 0.0419350 | - |
| 3 | 1 | 1.3200 | 0.0026562 | 0.0000000 | 0.0000000 | 0.023106 |
| 3 | 2 | 0.2772 | 0.0715960 | 0.0000000 | 0.0000000 | - |
| 4 | 1 | 1.4389 | 0.0103630 | 0.0074527 | 0.0030173 | 0.017101 |
| 4 | 2 | 0.3638 | 0.0914080 | 0.1323600 | 0.0535870 | - |
| 5 | 1 | 1.4381 | 0.0100030 | 0.0061908 | 0.0025064 | 0.017290 |
| 5 | 2 | 0.3665 | 0.0848280 | 0.1035800 | 0.0419350 | - |
| 6 | 1 | 1.4385 | 0.0101320 | 0.0064285 | 0.0026026 | 0.017192 |
| 6 | 2 | 0.3665 | 0.0873140 | 0.1091100 | 0.0441740 | - |
| 7 | 1 | 1.4389 | 0.0101650 | 0.0061908 | 0.0025064 | 0.017125 |
| 7 | 2 | 0.3679 | 0.0880240 | 0.1035800 | 0.0419350 | - |
| 8 | 1 | 1.4393 | 0.0102940 | 0.0064285 | 0.0026026 | 0.017027 |
| 8 | 2 | 0.3680 | 0.0905100 | 0.1091100 | 0.0441740 | - |

Diffusion coefficients are in cm; cross sections are in cm^-1. The reference
solution uses a P6 Legendre expansion.

| Method | SP3 DoFs per group | k_eff | Delta (pcm) | Max power error (%) |
| --- | ---: | ---: | ---: | ---: |
| CG Global | 33,442 / 33,442 | 1.025622 | 51.2 | 4.49 |
| DG Global | 74,016 / 32,896 | 1.025582 | 47.2 | 4.06 |
| DG Adaptive | 30,600 / 13,600 | 1.025487 | 37.7 | 2.72 |
| DG Goal | 22,122 / 22,122 | 1.025615 | 50.5 | 4.57 |
| Reference | - | 1.025110 | 0 | - |

For this case, adaptive DG reduces both the number of DoFs and the maximum
power error compared with the globally refined DG run. The assembly-averaged
power distribution was evaluated with DG-FEM and adaptive spatial refinement,
and scalar-flux distributions were exported for both energy groups.

### 2D IAEA LWR

The 2D IAEA benchmark is a simplified two-group light-water reactor problem
with 177 homogenized fuel assemblies. Each assembly is 20 cm by 20 cm, and the
external boundary condition is vacuum. The benchmark contains a strong
perturbation due to absorbent rods and large thermal-flux gradients near the
core-reflector interface.

Cross-section data for the 2D IAEA benchmark:

| Zone | Group | D | sigma_a | nu sigma_f | sigma_s,g1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 1.5 | 0.010120 | 0.0 | 0.0 |
| 1 | 2 | 0.4 | 0.080032 | 0.135 | 0.02 |
| 2 | 1 | 1.5 | 0.010120 | 0.0 | 0.0 |
| 2 | 2 | 0.4 | 0.085032 | 0.135 | 0.02 |
| 3 | 1 | 1.5 | 0.010120 | 0.0 | 0.0 |
| 3 | 2 | 0.4 | 0.130032 | 0.135 | 0.02 |
| 4 | 1 | 2.0 | 0.000160 | 0.0 | 0.0 |
| 4 | 2 | 0.3 | 0.010024 | 0.0 | 0.04 |

Diffusion coefficients are in cm; cross sections are in cm^-1. The fission
spectrum is `chi_1 = 1.0`, `chi_2 = 0.0`. The reference solution is a
LABAN-PEL P6 calculation with a nodal expansion method.

| Method | SP3 DoFs per group | k_eff | Delta (pcm) | Max power error (%) |
| --- | ---: | ---: | ---: | ---: |
| CG Global | 31,394 / 31,394 | 1.030061 | 47.6 | 2.79 |
| DG Global | 69,408 / 69,408 | 1.030053 | 46.8 | 2.73 |
| DG Adaptive | 36,608 / 20,592 | 1.030047 | 46.2 | 2.72 |
| DG Goal | 22,536 / 22,536 | 1.030041 | 45.6 | 2.71 |
| Reference | - | 1.029585 | 0 | - |

The global DG and adaptive DG results are close in eigenvalue and power error,
while goal-oriented refinement reaches the smallest listed DoF count and the
lowest maximum power error in this table. The assembly-averaged power
distribution and both group scalar fluxes were exported for visualization.

### 3D IAEA PWR

The 3D IAEA PWR benchmark is a standard neutronics validation problem. The core
contains 177 fuel assemblies and 64 reflector assemblies. The radial assembly
width is 20 cm, each assembly is 20 cm high, and the total core height is
340 cm. Albedo vacuum boundary conditions are imposed at the external
boundary.

Macroscopic cross sections for the 3D IAEA reactor:

| Material | D1 | D2 | sigma_a1 | sigma_a2 | sigma_s12 | nu sigma_f1 | nu sigma_f2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 - Fuel | 1.50 | 0.40 | 0.010 | 0.085 | 0.020 | 0.00 | 0.135 |
| 2 - Rodded Fuel | 1.50 | 0.40 | 0.010 | 0.130 | 0.020 | 0.00 | 0.135 |
| 3 - Exterior Fuel | 1.50 | 0.40 | 0.010 | 0.080 | 0.020 | 0.00 | 0.135 |
| 4 - Reflector | 2.00 | 0.30 | 0.000 | 0.010 | 0.040 | 0.00 | 0.000 |
| 5 - Rodded Reflector | 2.00 | 0.30 | 0.000 | 0.055 | 0.040 | 0.00 | 0.000 |

Diffusion coefficients are in cm; cross sections are in cm^-1. The reference
value is a VENTURE finite-difference calculation. FEMFFUSION is included as an
additional finite element comparison.

| Method | SP3 DoFs per group | k_eff | Delta (pcm) |
| --- | ---: | ---: | ---: |
| CG Global | 127,624,562 / 37,967,010 | 1.02955 | 52 |
| DG Global | 300,089,344 / 126,600,192 | 1.02955 | 52 |
| DG Adaptive | 25,528,064 / 10,769,652 | 1.02953 | 50 |
| DG Goal | 3,184,704 / 3,184,704 | 1.02957 | 54 |
| FEMFFUSION | 527,104 / 527,104 | 1.02971 | 68 |
| Reference | - | 1.02903 | 0 |

The 3D results show the value of matrix-free operator evaluation: the solver
can run problems with very large DoF counts without assembling the full active
system matrix. The adaptive and goal-oriented runs reduce the number of DoFs
substantially while preserving eigenvalue accuracy at the level shown in the
table.

For the final globally refined timing example, the fast and thermal groups
have 127,624,562 and 37,967,010 DoFs, with `p_fast = 3` and `p_thermal = 2`.
Starting from the interpolated solution of the previous cycle, the solver
requires 21 power iterations. The timing split is:

| Section | Wall time (s) | Fraction of total (%) |
| --- | ---: | ---: |
| Setup groups | 34.30 | 29.8 |
| Solve systems | 64.92 | 56.5 |
| Output mesh | 9.55 | 8.31 |
| Compute RHS | 6.74 | 5.86 |

The first refinement cycle requires roughly 340 power iterations. This makes
the eigensolver a natural target for improvement: a Krylov-Schur or Lanczos
method could reduce the number of outer iterations compared with the current
power method.

## deal.II Matrix-Free AMR Notes

The files in `deal.II_AMR/` contain the modified `deal.II` matrix-free sources
needed to run DG simulations with adaptive mesh refinement. The issue is that
matrix-free DG face integration, SIMD batching, and hanging-node topology
conflict with each other in the unmodified framework.

In DG-FEM, weak continuity is imposed through numerical fluxes on internal
faces. For performance, `deal.II` groups faces into SIMD batches in a way that
is analogous to cell batching for volume integrals. Adaptive refinement breaks
the simple one-to-one relation between the two cells adjacent to a face: a
coarse cell can face multiple fine cells. Supporting this requires three
changes at the same time:

- face batches must be constructed so vectorization remains valid;
- basis functions on the large cell must be evaluated consistently on
  quadrature points associated with the small-cell subfaces, with the required
  geometric projections;
- face and DoF data must remain contiguous and predictable enough for the
  matrix-free loop, MPI communication, and vector zeroing machinery.

### Creating Faces and Partitions

Face creation is handled by `generate_faces` in `face_setup_internal.h`. This
method fills the vectors `inner_faces`, `boundary_faces`, `inner_ghost_faces`,
and `refinement_edge_faces`.

For local internal and boundary faces, the partition vectors stored in
`TaskInfo` are resized like the cell partitions. This improves data locality:
the thread that processes a cell also processes the faces assigned to the same
partition. In the original implementation this was not done for
`inner_ghost_faces` and `refinement_edge_faces`, which were grouped into one
partition. Hanging-node faces were effectively treated as multigrid transfer
objects instead of as normal DG face-integration work.

The modified implementation changes the refined-face creation call so that
`number_cell_exterior` is set consistently with internal, ghost, and boundary
face creation. It also partitions `refinement_edge_faces` analogously to
internal and boundary faces. A further categorization change is required
because refined faces were recognized only when marked as
`FaceCategory::multigrid_refinement_edge`, which normally depends on the
presence of an `mg_level`. Without this change, `deal.II` can silently ignore
refined faces in non-multigrid matrix-free DG loops.

### Batches Construction

Face batches are built by `collect_faces_vectorization` in
`face_setup_internal.h`, using `compare_faces_for_vectorization` to decide
whether two faces are compatible. Compatibility depends on the local face
number, polynomial degree in p-adaptive cases, and `subface_index`. The
`subface_index` is valid on hanging-node faces and identifies which subface is
being integrated, for example the lower or upper half in 2D.

```cpp
if (face1.subface_index != face2.subface_index)
  return false;
```

The vectorization algorithm first groups faces into sets with compatible
metadata, then tries to pack each set into SIMD macro-faces. It prefers
contiguous memory chunks, then falls back to sparse gather batches, and finally
pads remaining lanes when a hard partition boundary or the last partition is
reached. If a boundary is soft, the leftover faces can be carried into the next
partition.

This keeps vectorized batches valid even with hanging nodes, because only
identical subface positions are grouped together. The price is that SIMD lane
occupation can be lower and padding can increase, but this is preferable to
mixing incompatible subfaces inside the same vectorized face batch.

### Global Faces Array Construction

`MatrixFree::initialize_indices` in `matrix_free.templates.h` calls
`generate_faces` when the `MatrixFree` object is initialized. Once
`refinement_edge_face_partition_data` is partitioned like
`cell_partition_data` and `face_partition_data`, the same
`hard_vectorization_boundary` logic used for internal and boundary faces must
also be applied to refinement-edge faces.

The original global face array order was:

```text
inner_faces
boundary_faces
inner_ghost_faces
refinement_edge_faces
```

The modified order moves local refinement-edge faces before ghost faces:

```text
inner_faces
boundary_faces
refinement_edge_faces
inner_ghost_faces
```

This keeps vectorized batches belonging purely to the local MPI process
adjacent in memory. It also fixes a lookup-table issue. The table
`cell_and_face_to_plain_faces` maps `(cell batch, local face number, SIMD lane)`
to the plain face index in the vectorized face array:

```cpp
TableIndices<3> index(face_info.faces[f].cells_interior[v] /
                      VectorizedArrayType::size(),
                      face_info.faces[f].interior_face_no,
                      face_info.faces[f].cells_interior[v] %
                      VectorizedArrayType::size());
face_info.cell_and_face_to_plain_faces(index) =
  f * VectorizedArrayType::size() + v;
```

In the original ordering, the loop filling this lookup stopped at the end of
the ghost-face range and excluded hanging-node faces. After moving
`refinement_edge_faces` before ghost faces, `ghost_face_partition_data.back()`
points to the absolute end of the global face array and the AMR faces are
included. The resize of `cell_level_index`, used later to preallocate geometry
data, is also corrected by replacing a hardcoded `[1]` access with `.back()`,
so the allocation covers all AMR-generated chunks.

### Batches Categorization

After the global face array is built, `initialize_indices` calls
`compute_face_index_compression`. This routine analyzes vectorized batches and
classifies their memory access patterns: contiguous, interleaved, sparse,
padded, and related cases.

The original logic was tightly tied to the old face-vector order: internal,
boundary, ghost, then refined faces. Refined faces were not properly included
when sizing the data structures used to classify batches containing external
cells. The modified implementation makes the routine agnostic with respect to
the exact face-vector structure. This slightly oversizes some arrays and
increases the number of calls to the `face_computation` lambda, but only by a
negligible amount proportional to the extra ghost faces. The old
`hold_all_faces_to_owned_cells` parameter becomes unused and is ignored.

### Data Races

Hanging-node face integration can create data races in threaded execution. If
several fine cells share a face with one coarse cell, multiple threads can
write to DoFs associated with the same coarse-side face.

Conflict management is handled with graph coloring. If two cells may write to
the same face DoFs, an edge is inserted in an adjacency graph stored with a
`DynamicSparsityPattern`; cells connected by an edge receive different colors.
The matrix-free loop then processes one color at a time.

The original check considered only cells on the same refinement level. This
fails in a coarse-fine situation: if parent cell `A` is refined into `A1` and
`A2`, and both children are adjacent to cell `B`, the children can detect the
dependency on `B`; but `B` detects the non-active parent `A`, which is then
discarded. The modified algorithm fills the sparsity pattern with 1-to-N
dependencies by checking active children on subfaces:

```text
for each cell:
  clear new_indices
  for each face of cell:
    if face is not boundary and neighbor is on the same MPI process:
      neighbor = cell.neighbor(face)

      if neighbor is active in the local map:
        add neighbor index

      if neighbor has children:
        for each child on the subface:
          if child is active in the local map:
            add child index

  sort new_indices
  add entries to the connectivity graph
```

This ensures that coarse-fine conflicts are colored correctly.

### MPI Communication

The original matrix-free MPI exchange setup used
`compute_tight_partitioners` in `dof_info.cc` to communicate only the DoFs
strictly required by the local face and cell work. For AMR support this had two
blocking issues.

First, `compute_tight_partitioners` did not receive the
`refinement_edge_faces` array, so dependencies introduced by refined faces were
not included in the communication pattern. The method signature was extended
to include refined faces.

Second, dependencies were mapped with a `Table<2, unsigned int>` indexed by
cell ID and local face number. With AMR, multiple fine cells can share the same
table index from the coarse-cell point of view, so entries were overwritten and
only the last child dependency survived. The table-based mapping was replaced
by a direct scan of the face vector, which naturally includes 1-to-N
coarse-fine dependencies.

### Vector Zeroing

After network exchanges are configured in `MatrixFree::initialize_indices`,
`DoFInfo::compute_vector_zero_access_pattern` in `dof_info.templates.h`
constructs the schedule used to zero destination vectors efficiently. This is
done in a distributed, threaded way instead of simply zeroing the whole vector
sequentially before the operation.

The original zeroing pattern did not account for `refinement_edge_faces`. As a
result, some DoFs touched by refined-face flux integration could remain
uninitialized, and worker contributions would be added to stale memory. The
modified code extends the zeroing pattern with the contribution of refined
faces.

### Execution of Refinement-Edge Faces

The indexing and communication changes are not sufficient unless the
matrix-free loop actually executes the user kernels on refinement-edge faces.
`MatrixFree::loop` calls the `TaskInfo::loop` implementation, and the work is
delegated to `MFWorker` objects.

The original `MFWorkerInterface` exposed user callbacks for cells, internal
faces, and boundary faces only. AMR support adds an analogous callback for
refinement boundaries. `TaskInfo::loop` then calls this callback whenever
`refinement_edge_face_partition_data` is non-empty. The same call is added to
the TBB multithreaded loop, in the same style as the existing internal-face and
boundary-face calls.
