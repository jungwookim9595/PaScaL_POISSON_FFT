# MPMC static Poisson-operator TDMA A/B

This package keeps the validated direct C2I-to-FFT, cuDecomp/NCCL, Stage-4
two-RHS path and adds reusable factors for the invariant P-P-N Poisson
operator.

## What changed

The dynamic control path still rebuilds and factorizes the tridiagonal
operator on every solve:

1. local modified Thomas for coefficients and two RHS arrays;
2. coefficient collectives for A, B, and C;
3. packed two-RHS collective;
4. reduced-system Thomas factorization/solve;
5. packed two-solution collective.

The static path performs the coefficient-only work once in
`cuda_Poisson_TDMA_static_initial()`, before the simulation loop:

- cache local modified-Thomas factors;
- communicate A, B, and C once;
- cache the transposed reduced operator;
- factor the reduced tridiagonal systems once.

Each later Poisson solve performs only:

- apply cached local factors to the new real/imaginary RHS pair;
- one packed RHS collective;
- apply cached reduced factors;
- one packed solution collective;
- update the local solution.

The distributed repeated-solve collective count is therefore reduced from
five to two.  The A/B selector is:

```bash
POISSON_TDMA_STATIC_OPERATOR=0  # dynamic control
POISSON_TDMA_STATIC_OPERATOR=1  # static operator reuse
```

The Poisson lower/upper diagonals depend only on the z row.  The static plan
therefore avoids two additional full 3D factor arrays: it stores two z vectors
plus two spectral planes for the row-1 correction.  The run log prints this
compact per-rank memory cost.

## Build and submit on MPMC

From the extracted project directory:

```bash
chmod +x ./*.sh
./build_and_submit_mpmc_static_tdma.sh
```

The scripts resolve the project from `PROJECT_ROOT`, `SLURM_SUBMIT_DIR`, or
their own location.  They never use `/var/spool/slurm/...` as the source tree.

One allocation runs:

- NP1: `1 x 1 x 1`;
- NP2: `1 x 1 x 2`;
- NP4: `1 x 2 x 2`;
- dynamic and static TDMA operator paths;
- debug, repeated unprofiled performance, and coarse phase runs;
- all three two-node NP4 rank pairings.

Full solution files are disabled.  RMS is evaluated once for the final
solution and remains outside all solve timers.

Results are written under:

```text
results_static_tdma/job_<SLURM_JOB_ID>/
```

The key generated files are:

- `summary.txt`;
- `static_tdma_ab_comparison.csv`;
- `tdma_phase_ab_comparison.csv`;
- `coarse_phase_comparison.csv`;
- debug/performance logs for every NP, placement, and operator variant.

## Acceptance checks

The static path is accepted only if:

1. dynamic and static final RMS values match within the summary tolerance;
2. logs state that the operator was prepared before timed solves;
3. debug logs report two repeated-solve TDMA collectives;
4. static median seconds/solve improves over dynamic;
5. the best NP4 time is compared directly with NP1.

This implementation targets a fixed grid, boundary conditions, and
constant-coefficient pressure operator.  If the mesh, boundary conditions,
or operator coefficients change, the static plan must be destroyed and
prepared again.
