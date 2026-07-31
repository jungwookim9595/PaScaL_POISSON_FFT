# Stage-4 implementation manifest

## Numerical and communication path

- `PaScaL_TDMA/src/tdmas_cuda.f90`
  - added a non-cyclic two-RHS Thomas kernel sharing one factorization.
- `PaScaL_TDMA/src/pascal_tdma_cuda.f90`
  - added two-RHS reduced workspaces and peer-contiguous communication buffers;
  - added batched modified-Thomas and solution-update kernels;
  - added forward and reverse two-RHS pack/unpack kernels;
  - added a five-collective distributed two-RHS API;
  - added debug labels, collective count, and extra plan-memory reporting.
- `src/fft_3d_poisson.f90`
  - added `POISSON_TDMA_2RHS` runtime A/B selection;
  - routed real and imaginary Fourier RHS arrays through one shared operator;
  - removed the duplicate imaginary coefficient workspaces in Stage-4 mode;
  - retained the original path for control runs and cyclic Z fallback.

## Current MPMC workflow

The Stage-4 numerical path is retained and fixed on during the new C2I
fusion/rank-placement experiment.

- `build_c2i_ab_mpmc.sh`: four-way legacy/direct debug/performance build.
- `run_mpmc_c2i_rank_sweep.slurm`: NP1/NP2 plus exhaustive NP4 pair placement.
- `summarize_c2i_rank_sweep.sh`: median A/B, coarse phase, RMS, and NP4 target report.
- `build_and_submit_mpmc_c2i_rank.sh`: one-command validation/build/submission.
- `validate_c2i_rank_static.sh`: source, scripts, decomposition, and timing checks.

## Validation completed before packaging

- all shell scripts pass `bash -n`;
- all relevant preprocessor branches pass `cpp`;
- the peer-contiguous two-RHS buffer layout passes a modeled 2- and 4-rank
  all-to-all round trip;
- the distributed modified-Thomas equations were checked against a direct
  tridiagonal reference for two independent RHS arrays;
- the summary script was exercised with existing NP1/NP2/NP4 logs;
- full NVHPC/CUDA compilation and runtime validation remain intentionally
  delegated to the MPMC cluster.
