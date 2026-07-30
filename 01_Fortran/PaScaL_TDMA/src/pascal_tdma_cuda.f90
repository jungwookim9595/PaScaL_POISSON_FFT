!======================================================================================================================
!> @file        pascal_tdma_cuda.cuf
!> @brief       PaScaL_TDMA - Parallel and Scalable Library for TriDiagonal Matrix Algorithm
!> @details     PaScaL_TDMA includes a CUDA implementation of PaScaL_TDMA, which accelerates 
!>              to solve many tridiagonal systems in multi-dimensional partial differential equations on GPU.
!>              It adopts the pipeline copy within the shared memory for the forward elemination and 
!>              backward substitution procudures of TDMA to reduce global memory access.
!>              For the main algorithm of PaScaL_TDMA, see also https://github.com/MPMC-Lab/PaScaL_TDMA.
!> 
!> @author      
!>              - Mingyu Yang (yang926@yonsei.ac.kr), School of Mathematics and Computing (Computational Science and Engineering), Yonsei University
!>              - Ji-Hoon Kang (jhkang@kisti.re.kr), Korea Institute of Science and Technology Information
!>              - Ki-Ha Kim (k-kiha@yonsei.ac.kr), School of Mathematics and Computing (Computational Science and Engineering), Yonsei University
!>              - Jung-Il Choi (jic@yonsei.ac.kr), School of Mathematics and Computing (Computational Science and Engineering), Yonsei University
!>
!> @date        May 2023
!> @version     2.0
!> @par         Copyright
!>              Copyright (c) 2019-2023 Mingyu Yang, Ki-Ha Kim and Jung-Il choi, Yonsei University and 
!>              Ji-Hoon Kang, Korea Institute of Science and Technology Information, All rights reserved.
!> @par         License     
!>              This project is release under the terms of the MIT License (see LICENSE file).
!======================================================================================================================

!>
!> @brief       Module for PaScaL_TDMA library with CUDA.
!> @details     It contains plans for tridiagonal systems of equations and subroutines for solving them 
!>              using the defined plans. The operation of the library includes the following three phases:
!>              (1) Create a data structure called a plan with the information for communication and reduced systems.
!>              (2) Solve the tridiagonal systems of equations executing from Step 1 to Step 5
!>              (3) Destroy the created plan
!>
module PaScaL_TDMA_cuda

    use mpi
    use tdma
    use cudafor

    implicit none

    !> @brief   Execution plan for many tridiagonal systems of equations.
    !> @details It uses MPI_Ialltoall function to distribute the modified tridiagonal systems to MPI processes
    !>          and build the reduced tridiagonal systems of equations. Currently it supports the equal size of domains.
	!>          It does not use derived datatypes.
    
    type, public :: ptdma_plan_many_cuda

        private

        integer :: ptdma_world      !< Single dimensional subcommunicator to assemble data for the reduced TDMA
        integer :: nprocs     		!< Communicator size of ptdma_world

        integer :: nx_sys           !< Number of tridiagonal systems of equations in x-direction per process
        integer :: ny_sys           !< Number of tridiagonal systems of equations in y-direction per process
        integer :: nz_row           !< Row size of partitioned tridiagonal matrix in z-direction per process

        integer :: nx_sys_rt        !< Number of tridiagonal systems to be solved in each process after transpose
        integer :: nz_row_rt        !< Number of rows of a reduced tridiagonal systems after transpose
        integer :: nz_row_rd        !< Number of rows of a reduced tridiagonal systems before transpose, 2.

        !> @{ Coefficient arrays after reduction, a: lower, b: diagonal, c: upper, d: rhs.
        !>    The orginal dimension (m:n) is reduced to (m:2)
        !>    The arrays are allocated in the device memory.
        double precision, allocatable, device, dimension(:,:,:) :: a_rd_d, b_rd_d, c_rd_d, d_rd_d
        double precision, allocatable, device, dimension(:,:,:) :: d2_rd_d
        !> @}

        !> @{ Coefficient arrays for cyclic TDMA in case of nprocs = 1.
        !>    The arrays are allocated in the device memory.
        double precision, allocatable, device, dimension(:,:,:) :: e_buff
        !> @}

        !> @{ Coefficient arrays after transpose of reduced systems, a: lower, b: diagonal, c: upper, d: rhs
        !>    The reduced dimension (m:2) changes to (m/np: 2*np) after transpose.
        !>    The arrays are allocated in the device memory.
        double precision, allocatable, device, dimension(:,:,:) :: a_rt_d, b_rt_d, c_rt_d, d_rt_d, e_rt_d
        double precision, allocatable, device, dimension(:,:,:) :: d2_rt_d
        !> @}

        !> @{ Buffers allocated in device memory for all-to-all communication
        double precision, allocatable, device, dimension(:) :: sendbuf, recvbuf
        double precision, allocatable, device, dimension(:) :: sendbuf_2rhs, recvbuf_2rhs
        !> @}

        ! Static-operator factors.  For a fixed tridiagonal operator the local
        ! modified-Thomas elimination and the reduced-system factorization are
        ! prepared once.  Every later solve applies these factors to new RHS
        ! arrays and communicates only the packed RHS and packed solution.
        ! For the Poisson operator, the original lower/upper coefficients are
        ! functions of z only and are shared by all spectral modes.  Store them
        ! once as row vectors; only the row-1 correction varies by mode.
        double precision, allocatable, device, dimension(:) :: static_lower_row_d
        double precision, allocatable, device, dimension(:) :: static_upper_row_d
        double precision, allocatable, device, dimension(:,:) :: static_row1_inverse_d
        double precision, allocatable, device, dimension(:,:) :: static_row1_upper_d
        logical :: static_operator_ready = .false.

        !> @{ Threads and blocks for CUDA kernel
        type(dim3)  :: threads, blocks, blocks_rt, blocks_alltoall
        !> @

        integer     :: shared_buffer_size   !< shared buffer size
    
    end type ptdma_plan_many_cuda

    ! Lightweight internal profiler for the distributed PaScaL_TDMA path.
    ! Keep this in lockstep with the Poisson profiler.  Only the debug build
    ! defines POISSON_DETAILED_PROFILE; the performance build therefore avoids
    ! every per-phase cudaDeviceSynchronize() in this module.
#ifdef POISSON_DETAILED_PROFILE
    logical, save :: ptdma_profile_enabled = .true.
#else
    logical, parameter :: ptdma_profile_enabled = .false.
#endif
    integer, parameter :: ptdma_profile_count = 19
    double precision, save :: ptdma_profile_times(ptdma_profile_count)
    integer, save :: ptdma_xy_call = 0
    integer, save :: ptdma_solve_call = 0
    logical, save :: ptdma_profile_2rhs = .false.
    logical, save :: ptdma_profile_static = .false.
    logical, save :: ptdma_2rhs_memory_reported = .false.
    private
    public  :: PaScaL_TDMA_plan_many_create_cuda
    public  :: PaScaL_TDMA_plan_many_destroy_cuda
    public  :: PaScaL_TDMA_many_solve_cuda
    public  :: PaScaL_TDMA_many_solve_cycle_cuda 
    public  :: PaScaL_TDMA_many_solve_2rhs_cuda
    public  :: PaScaL_TDMA_many_prepare_static_cuda
    public  :: PaScaL_TDMA_many_solve_static_2rhs_cuda
    public  :: PaScaL_TDMA_profile_set_enabled

    contains
    subroutine PaScaL_TDMA_profile_set_enabled(enabled, reset_counters)
        implicit none

        logical, intent(in) :: enabled
        logical, intent(in), optional :: reset_counters
        logical :: do_reset

        do_reset = .false.
        if (present(reset_counters)) do_reset = reset_counters

#ifdef POISSON_DETAILED_PROFILE
        ptdma_profile_enabled = enabled
#endif

        if (do_reset) then
            ptdma_profile_times = 0.0d0
            ptdma_xy_call = 0
            ptdma_solve_call = 0
            ptdma_profile_2rhs = .false.
            ptdma_profile_static = .false.
        endif
    end subroutine PaScaL_TDMA_profile_set_enabled


    subroutine ptdma_profile_start(t0)
        implicit none

        double precision, intent(out) :: t0
        integer :: ierr_cuda

#ifdef POISSON_DETAILED_PROFILE
        if (.not. ptdma_profile_enabled) then
            t0 = 0.0d0
            return
        endif
        ierr_cuda = cudaDeviceSynchronize()
        t0 = MPI_Wtime()
#else
        t0 = 0.0d0
#endif
    end subroutine ptdma_profile_start


    subroutine ptdma_profile_stop(t0, elapsed)
        implicit none

        double precision, intent(in)  :: t0
        double precision, intent(out) :: elapsed
        integer :: ierr_cuda

#ifdef POISSON_DETAILED_PROFILE
        if (.not. ptdma_profile_enabled) then
            elapsed = 0.0d0
            return
        endif
        ierr_cuda = cudaDeviceSynchronize()
        elapsed = MPI_Wtime() - t0
#else
        elapsed = 0.0d0
#endif
    end subroutine ptdma_profile_stop


    subroutine ptdma_profile_report()
        implicit none

        double precision :: tmin(ptdma_profile_count)
        double precision :: tmax(ptdma_profile_count)
        double precision :: tsum(ptdma_profile_count)
        double precision :: tavg(ptdma_profile_count)
        character(len=44) :: label
        integer :: ierr_mpi, myrank, nprocs, i

#ifndef POISSON_DETAILED_PROFILE
        return
#else
        if (.not. ptdma_profile_enabled) return

        call MPI_Comm_rank(MPI_COMM_WORLD, myrank, ierr_mpi)
        call MPI_Comm_size(MPI_COMM_WORLD, nprocs, ierr_mpi)

        call MPI_Reduce(ptdma_profile_times, tmin, &
                        ptdma_profile_count, MPI_DOUBLE_PRECISION, &
                        MPI_MIN, 0, MPI_COMM_WORLD, ierr_mpi)
        call MPI_Reduce(ptdma_profile_times, tsum, &
                        ptdma_profile_count, MPI_DOUBLE_PRECISION, &
                        MPI_SUM, 0, MPI_COMM_WORLD, ierr_mpi)
        call MPI_Reduce(ptdma_profile_times, tmax, &
                        ptdma_profile_count, MPI_DOUBLE_PRECISION, &
                        MPI_MAX, 0, MPI_COMM_WORLD, ierr_mpi)

        if (myrank /= 0) return

        tavg = tsum / dble(nprocs)

        write(*,'(/,A,I0)') &
            '================ PaScaL_TDMA profile call ', &
            ptdma_solve_call
        write(*,'(A)') &
            'Columns: rank-min / rank-avg / rank-max [seconds]'
        if (ptdma_profile_static) then
            write(*,'(A)') &
                '[PTDMA] Static operator: cached local/reduced factors + 2 RHS'
            write(*,'(A)') &
                '[PTDMA] Distributed CUDA-aware MPI collectives: 2'
        elseif (ptdma_profile_2rhs) then
            write(*,'(A)') &
                '[PTDMA] Stage-4 RHS mode: 2 RHS / shared operator'
            write(*,'(A)') &
                '[PTDMA] Distributed CUDA-aware MPI collectives: 5'
        else
            write(*,'(A)') &
                '[PTDMA] Legacy RHS mode: 1 RHS'
            write(*,'(A)') &
                '[PTDMA] Distributed CUDA-aware MPI collectives: 5 per RHS'
        endif

        do i = 1, ptdma_profile_count
            select case(i)
            case(1)
                label = '01 direct single-rank TDMA'
            case(2)
                if (ptdma_profile_static) then
                    label = '02 cached local 2-RHS reduction'
                else
                    label = '02 distributed modified Thomas'
                endif
            case(3)
                label = '03 xy2yz A pack kernel'
            case(4)
                label = '04 xy2yz A CUDA-aware MPI'
            case(5)
                label = '05 xy2yz A unpack kernel'
            case(6)
                label = '06 xy2yz B pack kernel'
            case(7)
                label = '07 xy2yz B CUDA-aware MPI'
            case(8)
                label = '08 xy2yz B unpack kernel'
            case(9)
                label = '09 xy2yz C pack kernel'
            case(10)
                label = '10 xy2yz C CUDA-aware MPI'
            case(11)
                label = '11 xy2yz C unpack kernel'
            case(12)
                if (ptdma_profile_2rhs) then
                    label = '12 xy2yz D(2RHS) pack kernel'
                else
                    label = '12 xy2yz D pack kernel'
                endif
            case(13)
                if (ptdma_profile_2rhs) then
                    label = '13 xy2yz D(2RHS) CUDA-aware MPI'
                else
                    label = '13 xy2yz D CUDA-aware MPI'
                endif
            case(14)
                if (ptdma_profile_2rhs) then
                    label = '14 xy2yz D(2RHS) unpack kernel'
                else
                    label = '14 xy2yz D unpack kernel'
                endif
            case(15)
                if (ptdma_profile_static) then
                    label = '15 cached reduced 2-RHS solve'
                else
                    label = '15 reduced-system TDMA'
                endif
            case(16)
                if (ptdma_profile_2rhs) then
                    label = '16 yz2xy solution(2RHS) pack'
                else
                    label = '16 yz2xy solution pack kernel'
                endif
            case(17)
                if (ptdma_profile_2rhs) then
                    label = '17 yz2xy solution(2RHS) MPI'
                else
                    label = '17 yz2xy solution CUDA-aware MPI'
                endif
            case(18)
                if (ptdma_profile_2rhs) then
                    label = '18 yz2xy solution(2RHS) unpack'
                else
                    label = '18 yz2xy solution unpack kernel'
                endif
            case(19)
                label = '19 distributed solution update'
            end select

            if (tmax(i) > 0.0d0) then
                write(*,'(A,1X,A44,3(1X,ES13.6))') &
                    '[PTDMA]', label, tmin(i), tavg(i), tmax(i)
            endif
        enddo

        write(*,'(A,/)') &
            '========================================================'
#endif
    end subroutine ptdma_profile_report

    !>
    !> @brief   Create a plan for many tridiagonal systems of equations.
    !> @param   p           Plan for a many tridiagonal system of equations
    !> @param   nx_sys      Number of tridiagonal systems of equations in x-direction per process
    !> @param   ny_sys      Number of tridiagonal systems of equations in y-direction per process
    !> @param   nz_row      Row size of partitioned tridiagonal matrix in z-direction per process
    !> @param   myrank      Rank ID in mpi_world
    !> @param   nprocs      Number of MPI process in mpi_world
    !> @param   mpi_world   Communicator for MPI_Gather and MPI_Scatter of reduced equations
    !>
    subroutine PaScaL_TDMA_plan_many_create_cuda(p, nx_sys, ny_sys, nz_row, myrank, nprocs, mpi_world, thread_in)

        implicit none
        
        type(ptdma_plan_many_cuda), intent(inout)  :: p
        type(dim3) :: thread_in

        integer, intent(in)     :: nx_sys
        integer, intent(in)     :: ny_sys
        integer, intent(in)     :: nz_row
        integer, intent(in)     :: myrank, nprocs, mpi_world

        integer :: nx_block, ny_block, nx_rt_block
        integer :: i, ierr
        integer :: ista, iend           ! First and last row indices of many tridiagonal systems of equations 
        integer :: nx_sys_rd, nz_row_rd ! Dimensions of many reduced tridiagonal systems
        integer :: nx_sys_rt, nz_row_rt ! Dimensions of many reduced tridiagonal systems after transpose
        logical :: plan_is_allocated, same_plan

        ! The Poisson FFT path may request the plan again on every CFD step.
        ! Reuse an identical plan, and safely replace it only when the actual
        ! redistributed spectral shape or communicator changes.
        plan_is_allocated = allocated(p%e_buff) .or. &
                            allocated(p%a_rd_d) .or. &
                            allocated(p%sendbuf)

        if (plan_is_allocated) then
            same_plan = p%nx_sys == nx_sys .and. &
                        p%ny_sys == ny_sys .and. &
                        p%nz_row == nz_row .and. &
                        p%nprocs == nprocs .and. &
                        p%ptdma_world == mpi_world .and. &
                        p%threads%x == thread_in%x .and. &
                        p%threads%y == thread_in%y

            if (same_plan) return

            call PaScaL_TDMA_plan_many_destroy_cuda(p)
        endif

        ! Assign plan variables and allocate coefficient arrays.
        p%nx_sys        = nx_sys
        p%ny_sys        = ny_sys
        p%nz_row        = nz_row
        p%ptdma_world   = mpi_world
        p%nprocs  = nprocs
    
        ! Specify dimensions for reduced systems.
        nx_sys_rd = nx_sys
        nz_row_rd = 2

        ! Specify dimensions for reduced systems after transpose.
        ! nx_sys_rt         : divide the number of tridiagonal systems of equations per each process  
        ! nx_sys_rt_array   : save the nx_sys_rt in nx_sys_rt_array for defining the DDT
        ! nz_row_rt         : dimensions of the reduced tridiagonal systems in the solving direction, nz_row_rd*nprocs
        call para_range(1, nx_sys_rd, nprocs, myrank, ista, iend)
        nx_sys_rt = iend - ista + 1
        nz_row_rt = nz_row_rd*nprocs
        
        ! Specify dimensions for reduced systems.
        p%nx_sys_rt     = nx_sys_rt
        p%nz_row_rt     = nz_row_rt
        p%nz_row_rd     = nz_row_rd

        ! Allocate coefficient arrays.
        if(p%nprocs.eq.1) then
            allocate( p%e_buff(nx_sys, ny_sys, nz_row) );  p%e_buff = 0.0d0
        else
            allocate( p%a_rd_d(nx_sys_rd, ny_sys, nz_row_rd) );  p%a_rd_d  = 0.0d0
            allocate( p%b_rd_d(nx_sys_rd, ny_sys, nz_row_rd) );  p%b_rd_d  = 0.0d0
            allocate( p%c_rd_d(nx_sys_rd, ny_sys, nz_row_rd) );  p%c_rd_d  = 0.0d0
            allocate( p%d_rd_d(nx_sys_rd, ny_sys, nz_row_rd) );  p%d_rd_d  = 0.0d0
            allocate( p%a_rt_d(nx_sys_rt, ny_sys, nz_row_rt) );  p%a_rt_d  = 0.0d0
            allocate( p%b_rt_d(nx_sys_rt, ny_sys, nz_row_rt) );  p%b_rt_d  = 0.0d0
            allocate( p%c_rt_d(nx_sys_rt, ny_sys, nz_row_rt) );  p%c_rt_d  = 0.0d0
            allocate( p%d_rt_d(nx_sys_rt, ny_sys, nz_row_rt) );  p%d_rt_d  = 0.0d0
            allocate( p%e_rt_d(nx_sys_rt, ny_sys, nz_row_rt) );  p%e_rt_d  = 0.0d0
            allocate( p%sendbuf(nx_sys*ny_sys*nz_row_rd))     ;  p%sendbuf = 0.0d0
            allocate( p%recvbuf(nx_sys*ny_sys*nz_row_rd))     ;  p%recvbuf = 0.0d0
        endif

        ! Check the thread configuration.
        ! Partial CUDA blocks are allowed; out-of-range threads are guarded in every kernel.
        if (thread_in%x <= 0 .or. thread_in%y <= 0) then
            if (myrank == 0) then
                print '(a,2(i0,1x))', '[Error] CUDA block dimensions must be positive: ', &
                                      thread_in%x, thread_in%y
            endif
            call MPI_Abort(mpi_world, 1, ierr)
        endif

        if (thread_in%x * thread_in%y > 1024) then
            if (myrank == 0) then
                print '(a,2(i0,1x),a,i0)', '[Error] CUDA block is too large: ', &
                                           thread_in%x, thread_in%y, &
                                           ' total threads = ', thread_in%x * thread_in%y
            endif
            call MPI_Abort(mpi_world, 1, ierr)
        endif

        ! Ceiling division: enough blocks are launched even when a dimension is not divisible
        ! by the requested CUDA block size. The final partial block is handled by bounds guards.
        nx_block    = (nx_sys    + thread_in%x - 1) / thread_in%x
        nx_rt_block = (nx_sys_rt + thread_in%x - 1) / thread_in%x
        ny_block    = (ny_sys    + thread_in%y - 1) / thread_in%y

        ! Define threads and blocks of dim3 type
        p%threads         = dim3( thread_in%x, thread_in%y,    1)
        p%blocks          = dim3( nx_block,    ny_block,       1)
        p%blocks_rt       = dim3( nx_rt_block, ny_block,       1)
        p%blocks_alltoall = dim3( nx_rt_block, ny_block,       nz_row_rd)

        ! Define the buffer size of shared memory for pipeline copy
        p%shared_buffer_size = kind(0.0d0)*(1+thread_in%x)*thread_in%y
        p%static_operator_ready = .false.

    end subroutine PaScaL_TDMA_plan_many_create_cuda

    subroutine PaScaL_TDMA_plan_many_ensure_2rhs_cuda(p)
        implicit none

        type(ptdma_plan_many_cuda), intent(inout) :: p

        if (p%nprocs == 1) return
        if (allocated(p%d2_rd_d)) return

        allocate(p%d2_rd_d(p%nx_sys, p%ny_sys, p%nz_row_rd))
        allocate(p%d2_rt_d(p%nx_sys_rt, p%ny_sys, p%nz_row_rt))
        allocate(p%sendbuf_2rhs(2*p%nx_sys*p%ny_sys*p%nz_row_rd))
        allocate(p%recvbuf_2rhs(2*p%nx_sys*p%ny_sys*p%nz_row_rd))
        p%d2_rd_d = 0.0d0
        p%d2_rt_d = 0.0d0
        p%sendbuf_2rhs = 0.0d0
        p%recvbuf_2rhs = 0.0d0

    end subroutine PaScaL_TDMA_plan_many_ensure_2rhs_cuda

    !>
    !> @brief   Destroy the allocated arrays in the defined plan_many.
    !> @param   p           Plan for many tridiagonal systems of equations
    !>
    subroutine PaScaL_TDMA_plan_many_destroy_cuda(p)
        implicit none

        type(ptdma_plan_many_cuda), intent(inout)  :: p

        if (allocated(p%e_buff)) deallocate(p%e_buff)

        if (allocated(p%a_rd_d)) deallocate(p%a_rd_d)
        if (allocated(p%b_rd_d)) deallocate(p%b_rd_d)
        if (allocated(p%c_rd_d)) deallocate(p%c_rd_d)
        if (allocated(p%d_rd_d)) deallocate(p%d_rd_d)
        if (allocated(p%d2_rd_d)) deallocate(p%d2_rd_d)

        if (allocated(p%a_rt_d)) deallocate(p%a_rt_d)
        if (allocated(p%b_rt_d)) deallocate(p%b_rt_d)
        if (allocated(p%c_rt_d)) deallocate(p%c_rt_d)
        if (allocated(p%d_rt_d)) deallocate(p%d_rt_d)
        if (allocated(p%d2_rt_d)) deallocate(p%d2_rt_d)
        if (allocated(p%e_rt_d)) deallocate(p%e_rt_d)

        if (allocated(p%sendbuf)) deallocate(p%sendbuf)
        if (allocated(p%recvbuf)) deallocate(p%recvbuf)
        if (allocated(p%sendbuf_2rhs)) deallocate(p%sendbuf_2rhs)
        if (allocated(p%recvbuf_2rhs)) deallocate(p%recvbuf_2rhs)
        if (allocated(p%static_lower_row_d)) deallocate(p%static_lower_row_d)
        if (allocated(p%static_upper_row_d)) deallocate(p%static_upper_row_d)
        if (allocated(p%static_row1_inverse_d)) deallocate(p%static_row1_inverse_d)
        if (allocated(p%static_row1_upper_d)) deallocate(p%static_row1_upper_d)

        p%nprocs = 0
        p%nx_sys = 0
        p%ny_sys = 0
        p%nz_row = 0
        p%static_operator_ready = .false.

    end subroutine PaScaL_TDMA_plan_many_destroy_cuda

    !>
    !> @brief   Solve many tridiagonal systems of equations.
    !> @param   p           Plan for many tridiagonal systems of equations
    !> @param   a_d         Coefficient array of lower diagonal elements
    !> @param   b_d         Coefficient array of diagonal elements
    !> @param   c_d         Coefficient array of upper diagonal elements
    !> @param   d_d         Coefficient array of right-hand side terms
    !>
    subroutine PaScaL_TDMA_many_solve_cuda(p, a_d, b_d, c_d, d_d)

        implicit none

        type(ptdma_plan_many_cuda), intent(inout)   :: p
        double precision, device, intent(inout)     :: a_d(:, :, :), b_d(:, :, :)
        double precision, device, intent(inout)     :: c_d(:, :, :), d_d(:, :, :)
        double precision :: prof_t0
        ptdma_xy_call = 0
        if (ptdma_profile_enabled) then
            ptdma_profile_times = 0.0d0
            ptdma_solve_call = ptdma_solve_call + 1
            ptdma_profile_2rhs = .false.
            ptdma_profile_static = .false.
        endif

        if(p%nprocs.eq.1) then
            ! Solve the tridiagonal system directly when nprocs = 1. 
            ! The size of shared memory is specified using shared_buffer_size
            call ptdma_profile_start(prof_t0)
            call tdma_many_cuda <<<p%blocks, p%threads, 6*p%shared_buffer_size>>> & ! 6 * 8 byte ?
                                (a_d, b_d, c_d, d_d, p%nx_sys, p%ny_sys, p%nz_row)
            call ptdma_profile_stop(prof_t0, ptdma_profile_times(1))
        else
            ! The modified Thomas algorithm
            ! The size of shared memory is specified using 'shared_buffer_size'
            call ptdma_profile_start(prof_t0)
            call PaScaL_TDMA_many_modified_Thomas_cuda <<<p%blocks, p%threads, 9*p%shared_buffer_size>>> &
                    (a_d, b_d, c_d, d_d, p%a_rd_d, p%b_rd_d, p%c_rd_d, p%d_rd_d, &
                     p%nx_sys, p%ny_sys, p%nz_row)
            call ptdma_profile_stop(prof_t0, ptdma_profile_times(2))

            ! Transpose the reduced systems of equations for TDMA using MPI_alltoall and CUDA-aware-MPI.
            call transpose_slab_xy_to_yz(p, p%a_rd_d, p%a_rt_d)
            call transpose_slab_xy_to_yz(p, p%b_rd_d, p%b_rt_d)
            call transpose_slab_xy_to_yz(p, p%c_rd_d, p%c_rt_d)
            call transpose_slab_xy_to_yz(p, p%d_rd_d, p%d_rt_d)

            ! Solve the reduced tridiagonal systems of equations using Thomas algorithm.
            ! The size of shared memory is specified using 'shared_buffer_size'
            call ptdma_profile_start(prof_t0)
            call tdma_many_cuda <<<p%blocks_rt, p%threads, 6*p%shared_buffer_size>>> &
                                (p%a_rt_d, p%b_rt_d, p%c_rt_d, p%d_rt_d, &
                                 p%nx_sys_rt, p%ny_sys, p%nz_row_rt)
            call ptdma_profile_stop(prof_t0, ptdma_profile_times(15))

            ! Transpose the obtained solutions to original reduced forms using MPI_Alltoall and CUDA-aware-MPI.
            call transpose_slab_yz_to_xy(p, p%d_rt_d, p%d_rd_d)

            ! Update solutions of the modified tridiagonal system with the solutions of the reduced tridiagonal system.
            ! The size of shared memory is specified using 'shared_buffer_size'
            call ptdma_profile_start(prof_t0)
            call PaScaL_TDMA_many_update_solution_cuda<<<p%blocks, p%threads, 2*p%shared_buffer_size>>> &
                                    (a_d, c_d, d_d, p%d_rd_d, &
                                     p%nx_sys, p%ny_sys, p%nz_row)
            call ptdma_profile_stop(prof_t0, ptdma_profile_times(19))
        endif
        call ptdma_profile_report()

    end subroutine PaScaL_TDMA_many_solve_cuda

    !>
    !> @brief Solve two RHS arrays that share the same non-cyclic operator.
    !> @details For nprocs > 1 this performs five collectives total:
    !>          A, B, C, packed(D1,D2), and packed(X1,X2).
    !>
    subroutine PaScaL_TDMA_many_solve_2rhs_cuda(p, a_d, b_d, c_d, d1_d, d2_d)

        implicit none

        type(ptdma_plan_many_cuda), intent(inout) :: p
        double precision, device, intent(inout) :: a_d(:, :, :), b_d(:, :, :)
        double precision, device, intent(inout) :: c_d(:, :, :)
        double precision, device, intent(inout) :: d1_d(:, :, :), d2_d(:, :, :)
        double precision :: prof_t0, extra_mib
        integer :: myrank, ierr_mpi

        ptdma_xy_call = 0
        if (ptdma_profile_enabled) then
            ptdma_profile_times = 0.0d0
            ptdma_solve_call = ptdma_solve_call + 1
            ptdma_profile_2rhs = .true.
            ptdma_profile_static = .false.
        endif

        if (p%nprocs.eq.1) then
            call ptdma_profile_start(prof_t0)
            call tdma_many_2rhs_cuda <<<p%blocks, p%threads, 8*p%shared_buffer_size>>> &
                                     (a_d, b_d, c_d, d1_d, d2_d, &
                                      p%nx_sys, p%ny_sys, p%nz_row)
            call ptdma_profile_stop(prof_t0, ptdma_profile_times(1))
        else
            call PaScaL_TDMA_plan_many_ensure_2rhs_cuda(p)
            if (.not. ptdma_2rhs_memory_reported) then
                call MPI_Comm_rank(MPI_COMM_WORLD, myrank, ierr_mpi)
                if (myrank == 0) then
                    ! d2_rd + d2_rt + two double-sized communication buffers.
                    extra_mib = dble(6 * p%nx_sys * p%ny_sys * p%nz_row_rd) &
                                * 8.0d0 / (1024.0d0 * 1024.0d0)
                    write(*,'(A,F10.3,A)') &
                        '[PTDMA] Stage-4 extra plan memory/rank: ', &
                        extra_mib, ' MiB'
                endif
                ptdma_2rhs_memory_reported = .true.
            endif

            call ptdma_profile_start(prof_t0)
            call PaScaL_TDMA_many_modified_Thomas_2rhs_cuda &
                <<<p%blocks, p%threads, 11*p%shared_buffer_size>>> &
                (a_d, b_d, c_d, d1_d, d2_d, &
                 p%a_rd_d, p%b_rd_d, p%c_rd_d, p%d_rd_d, p%d2_rd_d, &
                 p%nx_sys, p%ny_sys, p%nz_row)
            call ptdma_profile_stop(prof_t0, ptdma_profile_times(2))

            call transpose_slab_xy_to_yz(p, p%a_rd_d, p%a_rt_d)
            call transpose_slab_xy_to_yz(p, p%b_rd_d, p%b_rt_d)
            call transpose_slab_xy_to_yz(p, p%c_rd_d, p%c_rt_d)
            call transpose_slab_xy_to_yz_2rhs( &
                p, p%d_rd_d, p%d2_rd_d, p%d_rt_d, p%d2_rt_d)

            call ptdma_profile_start(prof_t0)
            call tdma_many_2rhs_cuda &
                <<<p%blocks_rt, p%threads, 8*p%shared_buffer_size>>> &
                (p%a_rt_d, p%b_rt_d, p%c_rt_d, p%d_rt_d, p%d2_rt_d, &
                 p%nx_sys_rt, p%ny_sys, p%nz_row_rt)
            call ptdma_profile_stop(prof_t0, ptdma_profile_times(15))

            call transpose_slab_yz_to_xy_2rhs( &
                p, p%d_rt_d, p%d2_rt_d, p%d_rd_d, p%d2_rd_d)

            call ptdma_profile_start(prof_t0)
            call PaScaL_TDMA_many_update_solution_2rhs_cuda &
                <<<p%blocks, p%threads, 4*p%shared_buffer_size>>> &
                (a_d, c_d, d1_d, d2_d, p%d_rd_d, p%d2_rd_d, &
                 p%nx_sys, p%ny_sys, p%nz_row)
            call ptdma_profile_stop(prof_t0, ptdma_profile_times(19))
        endif

        call ptdma_profile_report()

    end subroutine PaScaL_TDMA_many_solve_2rhs_cuda

    !>
    !> @brief Pre-factor a fixed non-cyclic operator for repeated two-RHS solves.
    !> @details For distributed systems, A/B/C are reduced and transposed once,
    !>          and both the local modified system and global reduced system are
    !>          factorized.  Subsequent solves communicate only RHS and solution.
    !>
    subroutine PaScaL_TDMA_many_prepare_static_cuda(p, a_d, b_d, c_d)

        implicit none

        type(ptdma_plan_many_cuda), intent(inout) :: p
        double precision, device, intent(inout) :: a_d(:, :, :), b_d(:, :, :)
        double precision, device, intent(inout) :: c_d(:, :, :)
        double precision :: factor_mib
        integer :: ierr_cuda, ierr_mpi, myrank

        if (p%static_operator_ready) return

        if (p%nprocs.eq.1) then
            call tdma_static_prepare_cuda<<<p%blocks, p%threads>>> &
                (a_d, b_d, c_d, p%nx_sys, p%ny_sys, p%nz_row)
        else
            call PaScaL_TDMA_plan_many_ensure_2rhs_cuda(p)

            if (.not. allocated(p%static_lower_row_d)) then
                allocate(p%static_lower_row_d(p%nz_row))
                allocate(p%static_upper_row_d(p%nz_row))
                allocate(p%static_row1_inverse_d(p%nx_sys, p%ny_sys))
                allocate(p%static_row1_upper_d(p%nx_sys, p%ny_sys))
            endif

            call PaScaL_TDMA_copy_shared_rows_cuda<<<1, 1>>> &
                (a_d, c_d, p%static_lower_row_d, p%static_upper_row_d, &
                 min(2, p%nx_sys), p%nz_row)
            call PaScaL_TDMA_many_prepare_modified_static_cuda &
                <<<p%blocks, p%threads>>> &
                (a_d, b_d, c_d, p%static_row1_inverse_d, &
                 p%static_row1_upper_d, &
                 p%a_rd_d, p%b_rd_d, p%c_rd_d, &
                 p%nx_sys, p%ny_sys, p%nz_row)

            ! These three coefficient collectives are initialization-only.
            ptdma_xy_call = 0
            call transpose_slab_xy_to_yz(p, p%a_rd_d, p%a_rt_d)
            call transpose_slab_xy_to_yz(p, p%b_rd_d, p%b_rt_d)
            call transpose_slab_xy_to_yz(p, p%c_rd_d, p%c_rt_d)

            call tdma_static_prepare_cuda<<<p%blocks_rt, p%threads>>> &
                (p%a_rt_d, p%b_rt_d, p%c_rt_d, &
                 p%nx_sys_rt, p%ny_sys, p%nz_row_rt)
        endif

        ierr_cuda = cudaDeviceSynchronize()
        if (ierr_cuda /= 0) then
            call MPI_Abort(p%ptdma_world, ierr_cuda, ierr_mpi)
        endif

        p%static_operator_ready = .true.

        call MPI_Comm_rank(MPI_COMM_WORLD, myrank, ierr_mpi)
        if (myrank == 0) then
            write(*,'(A)') &
                '[PTDMA-STATIC] Operator factors prepared before timed solves.'
            if (p%nprocs > 1) then
                factor_mib = dble(2*p%nz_row+2*p%nx_sys*p%ny_sys) &
                             * 8.0d0 / (1024.0d0*1024.0d0)
                write(*,'(A,F10.3,A)') &
                    '[PTDMA-STATIC] Compact factor memory/rank: ', &
                    factor_mib, ' MiB'
                write(*,'(A)') &
                    '[PTDMA-STATIC] Repeated-solve collectives: 2 (RHS, solution).'
            else
                write(*,'(A)') &
                    '[PTDMA-STATIC] Single-rank Thomas factors will be reused.'
            endif
        endif

    end subroutine PaScaL_TDMA_many_prepare_static_cuda


    !>
    !> @brief Apply a pre-factorized fixed operator to two new RHS arrays.
    !>
    subroutine PaScaL_TDMA_many_solve_static_2rhs_cuda( &
        p, a_d, b_d, c_d, d1_d, d2_d)

        implicit none

        type(ptdma_plan_many_cuda), intent(inout) :: p
        double precision, device, intent(inout) :: a_d(:, :, :), b_d(:, :, :)
        double precision, device, intent(inout) :: c_d(:, :, :)
        double precision, device, intent(inout) :: d1_d(:, :, :), d2_d(:, :, :)
        double precision :: prof_t0
        integer :: ierr_mpi

        if (.not. p%static_operator_ready) then
            write(*,'(A)') &
                '[PTDMA-STATIC-ERROR] Solve requested before operator prepare.'
            call MPI_Abort(p%ptdma_world, 91, ierr_mpi)
        endif

        ptdma_xy_call = 0
        if (ptdma_profile_enabled) then
            ptdma_profile_times = 0.0d0
            ptdma_solve_call = ptdma_solve_call + 1
            ptdma_profile_2rhs = .true.
            ptdma_profile_static = .true.
        endif

        if (p%nprocs.eq.1) then
            call ptdma_profile_start(prof_t0)
            call tdma_static_apply_2rhs_cuda<<<p%blocks, p%threads>>> &
                (a_d, b_d, c_d, d1_d, d2_d, &
                 p%nx_sys, p%ny_sys, p%nz_row)
            call ptdma_profile_stop(prof_t0, ptdma_profile_times(1))
        else
            call ptdma_profile_start(prof_t0)
            call PaScaL_TDMA_many_apply_modified_static_2rhs_cuda &
                <<<p%blocks, p%threads>>> &
                (b_d, p%static_lower_row_d, p%static_upper_row_d, &
                 p%static_row1_inverse_d, p%static_row1_upper_d, &
                 d1_d, d2_d, &
                 p%d_rd_d, p%d2_rd_d, &
                 p%nx_sys, p%ny_sys, p%nz_row)
            call ptdma_profile_stop(prof_t0, ptdma_profile_times(2))

            call transpose_slab_xy_to_yz_2rhs( &
                p, p%d_rd_d, p%d2_rd_d, p%d_rt_d, p%d2_rt_d)

            call ptdma_profile_start(prof_t0)
            call tdma_static_apply_2rhs_cuda<<<p%blocks_rt, p%threads>>> &
                (p%a_rt_d, p%b_rt_d, p%c_rt_d, p%d_rt_d, p%d2_rt_d, &
                 p%nx_sys_rt, p%ny_sys, p%nz_row_rt)
            call ptdma_profile_stop(prof_t0, ptdma_profile_times(15))

            call transpose_slab_yz_to_xy_2rhs( &
                p, p%d_rt_d, p%d2_rt_d, p%d_rd_d, p%d2_rd_d)

            call ptdma_profile_start(prof_t0)
            call PaScaL_TDMA_many_update_solution_2rhs_cuda &
                <<<p%blocks, p%threads, 4*p%shared_buffer_size>>> &
                (a_d, c_d, d1_d, d2_d, p%d_rd_d, p%d2_rd_d, &
                 p%nx_sys, p%ny_sys, p%nz_row)
            call ptdma_profile_stop(prof_t0, ptdma_profile_times(19))
        endif

        call ptdma_profile_report()

    end subroutine PaScaL_TDMA_many_solve_static_2rhs_cuda

    !>
    !> @brief   Solve many cyclic tridiagonal systems of equations.
    !> @param   p           Plan for many tridiagonal systems of equations
    !> @param   a_d         Coefficient array of lower diagonal elements
    !> @param   b_d         Coefficient array of diagonal elements
    !> @param   c_d         Coefficient array of upper diagonal elements
    !> @param   d_d         Coefficient array of right-hand side terms
    !>
    subroutine PaScaL_TDMA_many_solve_cycle_cuda(p, a_d, b_d, c_d, d_d)

        implicit none

        type(ptdma_plan_many_cuda), intent(inout)   :: p
        double precision, device, intent(inout)     :: a_d(:, :, :), b_d(:, :, :)
        double precision, device, intent(inout)     :: c_d(:, :, :), d_d(:, :, :)
        double precision :: prof_t0
        ptdma_xy_call = 0
        if (ptdma_profile_enabled) then
            ptdma_profile_times = 0.0d0
            ptdma_solve_call = ptdma_solve_call + 1
            ptdma_profile_2rhs = .false.
            ptdma_profile_static = .false.
        endif

        if(p%nprocs.eq.1) then
            ! Solve the cyclic tridiagonal system directly when nprocs = 1. 
            ! The size of shared memory is specified using shared_buffer_size
            call ptdma_profile_start(prof_t0)
            call tdma_cycl_many_cuda<<<p%blocks, p%threads, 8*p%shared_buffer_size>>> &
                                    (a_d, b_d, c_d, d_d, p%e_buff, &
                                     p%nx_sys, p%ny_sys, p%nz_row)
            call ptdma_profile_stop(prof_t0, ptdma_profile_times(1))
        else
            ! The modified Thomas algorithm
            ! The size of shared memory is specified using 'shared_buffer_size'
            call ptdma_profile_start(prof_t0)
            call PaScaL_TDMA_many_modified_Thomas_cuda<<<p%blocks, p%threads, 9*p%shared_buffer_size>>> &
                (a_d, b_d, c_d, d_d, p%a_rd_d, p%b_rd_d, p%c_rd_d, p%d_rd_d, &
                 p%nx_sys, p%ny_sys, p%nz_row)
            call ptdma_profile_stop(prof_t0, ptdma_profile_times(2))
 
            ! Transpose the reduced systems of equations for TDMA using MPI_alltoall and CUDA-aware-MPI.
            call transpose_slab_xy_to_yz(p, p%a_rd_d, p%a_rt_d)
            call transpose_slab_xy_to_yz(p, p%b_rd_d, p%b_rt_d)
            call transpose_slab_xy_to_yz(p, p%c_rd_d, p%c_rt_d)
            call transpose_slab_xy_to_yz(p, p%d_rd_d, p%d_rt_d)
    
            ! Solve the reduced tridiagonal systems of equations using Thomas algorithm.
            ! The size of shared memory is specified using 'shared_buffer_size'
            call ptdma_profile_start(prof_t0)
            call tdma_cycl_many_cuda<<<p%blocks_rt, p%threads, 8*p%shared_buffer_size>>> &
                                    (p%a_rt_d, p%b_rt_d, p%c_rt_d, p%d_rt_d, p%e_rt_d, &
                                     p%nx_sys_rt, p%ny_sys, p%nz_row_rt)
            call ptdma_profile_stop(prof_t0, ptdma_profile_times(15))

            ! Transpose the obtained solutions to original reduced forms using MPI_alltoall and CUDA-aware-MPI.
            call transpose_slab_yz_to_xy(p, p%d_rt_d, p%d_rd_d)

            ! Update solutions of the modified tridiagonal system with the solutions of the reduced tridiagonal system.
            ! The size of shared memory is specified using 'shared_buffer_size'
            call ptdma_profile_start(prof_t0)
            call PaScaL_TDMA_many_update_solution_cuda<<<p%blocks, p%threads, 2*p%shared_buffer_size>>> &
                                                    (a_d, c_d, d_d, p%d_rd_d, &
                                                     p%nx_sys, p%ny_sys, p%nz_row)
            call ptdma_profile_stop(prof_t0, ptdma_profile_times(19))
        endif
        call ptdma_profile_report()

    end subroutine PaScaL_TDMA_many_solve_cycle_cuda

    !>
    !> @brief   The modified Thomas algorithm : elimination of lower diagonal elements
    !> @param   a           Coefficient array of lower diagonal elements
    !> @param   b           Coefficient array of diagonal elements
    !> @param   c           Coefficient array of upper diagonal elements
    !> @param   d           Coefficient array of right-hand side terms
    !> @param   a_rd        Reduced coefficient array of lower diagonal elements
    !> @param   b_rd        Reduced coefficient array of diagonal elements
    !> @param   c_rd        Reduced coefficient array of upper diagonal elements
    !> @param   d_rd        Reduced doefficient array of right-hand side terms
    !> @param   nz_row      Row size of partitioned tridiagonal matrix in z-direction per process
    !>
    attributes(global) subroutine PaScaL_TDMA_many_modified_Thomas_cuda(a, b, c, d, a_rd, b_rd, c_rd, d_rd, &
                                                                          nx_sys, ny_sys, nz_row)

        implicit none

        integer, value, intent(in)      :: nx_sys, ny_sys, nz_row
        double precision, device, intent(inout) :: a(:, :, :), c(:, :, :), d(:, :, :)
        double precision, device, intent(in)    :: b(:, :, :)
        double precision, device, intent(inout) :: a_rd(:, :, :), b_rd(:, :, :)
        double precision, device, intent(inout) :: c_rd(:, :, :), d_rd(:, :, :)

        ! Temporary variables for computation
        integer :: i, j, k
        integer :: ti, tj, tk
        double precision :: r

        ! Block shared memory for pipeline copy
        double precision, shared :: a1(blockdim%x+1, blockdim%y), a0(blockdim%x+1, blockdim%y)
        double precision, shared :: b1(blockdim%x+1, blockdim%y), b0(blockdim%x+1, blockdim%y)
        double precision, shared :: c1(blockdim%x+1, blockdim%y), c0(blockdim%x+1, blockdim%y)
        double precision, shared :: d1(blockdim%x+1, blockdim%y), d0(blockdim%x+1, blockdim%y)
        double precision, shared :: r0(blockdim%x+1, blockdim%y)

        ! Global index
        j = (blockidx%x-1) * blockdim%x + threadidx%x
        k = (blockidx%y-1) * blockdim%y + threadidx%y

        ! Local index in block
        tj = threadidx%x
        tk = threadidx%y

        ! The final CUDA block can be partial after ceiling division.
        ! No synchronization is used in this kernel, so inactive threads can return safely.
        if (j > nx_sys .or. k > ny_sys) return

        ! The modified Thomas algorithm : elimination of lower diagonal elements. 
        ! First & second indices indicate a number of independent many tridiagonal systems for parallezation.
        ! Third index indicates a row number in a partitioned tridiagonal system .
		! Therefore, first & second indices are for thread IDs.
        ! The modified Thomas algorithm : elimination of lower diagonal elements. 

        a0(tj, tk) = a(j, k, 1)
        b0(tj, tk) = b(j, k, 1)
        c0(tj, tk) = c(j, k, 1)
        d0(tj, tk) = d(j, k, 1)

        a0(tj, tk) = a0(tj, tk) / b0(tj, tk)
        c0(tj, tk) = c0(tj, tk) / b0(tj, tk)
        d0(tj, tk) = d0(tj, tk) / b0(tj, tk)

        a(j, k,1) = a0(tj, tk)
        c(j, k,1) = c0(tj, tk)
        d(j, k,1) = d0(tj, tk)

        a1(tj, tk) = a(j, k, 2)
        b1(tj, tk) = b(j, k, 2)
        c1(tj, tk) = c(j, k, 2)
        d1(tj, tk) = d(j, k, 2)

        a1(tj, tk) = a1(tj, tk) / b1(tj, tk)
        c1(tj, tk) = c1(tj, tk) / b1(tj, tk)
        d1(tj, tk) = d1(tj, tk) / b1(tj, tk)

        a(j, k,2) = a1(tj, tk)
        c(j, k,2) = c1(tj, tk)
        d(j, k,2) = d1(tj, tk)

        do i = 3, nz_row

            ! Pipeline copy of (i-1)th data using shared memory
            a0(tj, tk) = a1(tj, tk)
            c0(tj, tk) = c1(tj, tk)
            d0(tj, tk) = d1(tj, tk)

            ! Load i-th data from global memory
            a1(tj, tk) = a(j, k,i)
            b1(tj, tk) = b(j, k,i)
            c1(tj, tk) = c(j, k,i)
            d1(tj, tk) = d(j, k,i)

            r0(tj, tk) =  1.0d0 / (b1(tj, tk) - a1(tj, tk) * c0(tj, tk))
            d1(tj, tk) =  r0(tj, tk) * (d1(tj, tk) - a1(tj, tk) * d0(tj, tk))
            c1(tj, tk) =  r0(tj, tk) * c1(tj, tk)
            a1(tj, tk) = -r0(tj, tk) * a1(tj, tk) * a0(tj, tk)

            ! Save updated i-th data to global memory
            a(j, k,i) = a1(tj, tk)
            c(j, k,i) = c1(tj, tk)
            d(j, k,i) = d1(tj, tk)

        enddo

        ! Construct many reduced tridiagonal systems per each rank. Each process has two rows of reduced systems.
        a_rd(j, k,2) = a1(tj, tk)
        b_rd(j, k,2) = 1.0d0
        c_rd(j, k,2) = c1(tj, tk)
        d_rd(j, k,2) = d1(tj, tk)

        ! Pipeline copy between arrays in shared memory
        a1(tj, tk) = a0(tj, tk)
        c1(tj, tk) = c0(tj, tk)
        d1(tj, tk) = d0(tj, tk)

        ! The modified Thomas algorithm : elimination of upper diagonal elements.
        do i = nz_row-2, 2, -1

            ! Load i-th data from global memory
            a0(tj, tk) = a(j, k,i)
            c0(tj, tk) = c(j, k,i)
            d0(tj, tk) = d(j, k,i)

            d0(tj, tk) = d0(tj, tk) - c0(tj, tk)*d1(tj, tk)
            a0(tj, tk) = a0(tj, tk) - c0(tj, tk)*a1(tj, tk)
            c0(tj, tk) =-c0(tj, tk) * c1(tj, tk)

            ! Pipeline copy of updated i-th data using shared memory
            a1(tj, tk) = a0(tj, tk)
            c1(tj, tk) = c0(tj, tk)
            d1(tj, tk) = d0(tj, tk)

            ! Save updated data to global memory
            a(j, k,i) = a0(tj, tk)
            c(j, k,i) = c0(tj, tk)
            d(j, k,i) = d0(tj, tk)
        enddo

        a0(tj, tk) = a(j, k,1)
        c0(tj, tk) = c(j, k,1)
        d0(tj, tk) = d(j, k,1)

        r0(tj, tk) = 1.0d0 / (1.0d0 - a1(tj, tk) * c0(tj, tk))
        d0(tj, tk) =  r0(tj, tk) * (d0(tj, tk) - c0(tj, tk) * d1(tj, tk))
        a0(tj, tk) =  r0(tj, tk) * a0(tj, tk)
        c0(tj, tk) = -r0(tj, tk) * c0(tj, tk) * c1(tj, tk)

        d(j, k,1) = d0(tj, tk)
        a(j, k,1) = a0(tj, tk)
        c(j, k,1) = c0(tj, tk)

        ! Construct many reduced tridiagonal systems per each rank. Each process has two rows of reduced systems.
        a_rd(j, k,1) = a0(tj, tk)
        b_rd(j, k,1) = 1.0d0
        c_rd(j, k,1) = c0(tj, tk)
        d_rd(j, k,1) = d0(tj, tk)
        

    end subroutine PaScaL_TDMA_many_modified_Thomas_cuda

    !>
    !> @brief Modified Thomas reduction for two RHS arrays and one operator.
    !>
    attributes(global) subroutine PaScaL_TDMA_many_modified_Thomas_2rhs_cuda( &
        a, b, c, d1, d2, a_rd, b_rd, c_rd, d1_rd, d2_rd, &
        nx_sys, ny_sys, nz_row)

        implicit none

        integer, value, intent(in) :: nx_sys, ny_sys, nz_row
        double precision, device, intent(inout) :: a(:, :, :), c(:, :, :)
        double precision, device, intent(inout) :: d1(:, :, :), d2(:, :, :)
        double precision, device, intent(in)    :: b(:, :, :)
        double precision, device, intent(inout) :: a_rd(:, :, :), b_rd(:, :, :)
        double precision, device, intent(inout) :: c_rd(:, :, :)
        double precision, device, intent(inout) :: d1_rd(:, :, :), d2_rd(:, :, :)

        integer :: i, j, k
        integer :: tj, tk
        double precision :: r

        double precision, shared :: a1(blockdim%x+1, blockdim%y), a0(blockdim%x+1, blockdim%y)
        double precision, shared :: b1(blockdim%x+1, blockdim%y), b0(blockdim%x+1, blockdim%y)
        double precision, shared :: c1(blockdim%x+1, blockdim%y), c0(blockdim%x+1, blockdim%y)
        double precision, shared :: d11(blockdim%x+1, blockdim%y), d10(blockdim%x+1, blockdim%y)
        double precision, shared :: d21(blockdim%x+1, blockdim%y), d20(blockdim%x+1, blockdim%y)
        double precision, shared :: r0(blockdim%x+1, blockdim%y)

        j = (blockidx%x-1) * blockdim%x + threadidx%x
        k = (blockidx%y-1) * blockdim%y + threadidx%y
        tj = threadidx%x
        tk = threadidx%y

        if (j > nx_sys .or. k > ny_sys) return

        a0(tj, tk)  = a(j, k, 1)
        b0(tj, tk)  = b(j, k, 1)
        c0(tj, tk)  = c(j, k, 1)
        d10(tj, tk) = d1(j, k, 1)
        d20(tj, tk) = d2(j, k, 1)

        r = 1.0d0 / b0(tj, tk)
        a0(tj, tk)  = r * a0(tj, tk)
        c0(tj, tk)  = r * c0(tj, tk)
        d10(tj, tk) = r * d10(tj, tk)
        d20(tj, tk) = r * d20(tj, tk)

        a(j, k, 1)  = a0(tj, tk)
        c(j, k, 1)  = c0(tj, tk)
        d1(j, k, 1) = d10(tj, tk)
        d2(j, k, 1) = d20(tj, tk)

        a1(tj, tk)  = a(j, k, 2)
        b1(tj, tk)  = b(j, k, 2)
        c1(tj, tk)  = c(j, k, 2)
        d11(tj, tk) = d1(j, k, 2)
        d21(tj, tk) = d2(j, k, 2)

        r = 1.0d0 / b1(tj, tk)
        a1(tj, tk)  = r * a1(tj, tk)
        c1(tj, tk)  = r * c1(tj, tk)
        d11(tj, tk) = r * d11(tj, tk)
        d21(tj, tk) = r * d21(tj, tk)

        a(j, k, 2)  = a1(tj, tk)
        c(j, k, 2)  = c1(tj, tk)
        d1(j, k, 2) = d11(tj, tk)
        d2(j, k, 2) = d21(tj, tk)

        do i = 3, nz_row
            a0(tj, tk)  = a1(tj, tk)
            c0(tj, tk)  = c1(tj, tk)
            d10(tj, tk) = d11(tj, tk)
            d20(tj, tk) = d21(tj, tk)

            a1(tj, tk)  = a(j, k, i)
            b1(tj, tk)  = b(j, k, i)
            c1(tj, tk)  = c(j, k, i)
            d11(tj, tk) = d1(j, k, i)
            d21(tj, tk) = d2(j, k, i)

            r0(tj, tk)  = 1.0d0 / (b1(tj, tk) - a1(tj, tk) * c0(tj, tk))
            d11(tj, tk) = r0(tj, tk) * (d11(tj, tk) - a1(tj, tk) * d10(tj, tk))
            d21(tj, tk) = r0(tj, tk) * (d21(tj, tk) - a1(tj, tk) * d20(tj, tk))
            c1(tj, tk)  = r0(tj, tk) * c1(tj, tk)
            a1(tj, tk)  = -r0(tj, tk) * a1(tj, tk) * a0(tj, tk)

            a(j, k, i)  = a1(tj, tk)
            c(j, k, i)  = c1(tj, tk)
            d1(j, k, i) = d11(tj, tk)
            d2(j, k, i) = d21(tj, tk)
        enddo

        a_rd(j, k, 2)  = a1(tj, tk)
        b_rd(j, k, 2)  = 1.0d0
        c_rd(j, k, 2)  = c1(tj, tk)
        d1_rd(j, k, 2) = d11(tj, tk)
        d2_rd(j, k, 2) = d21(tj, tk)

        a1(tj, tk)  = a0(tj, tk)
        c1(tj, tk)  = c0(tj, tk)
        d11(tj, tk) = d10(tj, tk)
        d21(tj, tk) = d20(tj, tk)

        do i = nz_row-2, 2, -1
            a0(tj, tk)  = a(j, k, i)
            c0(tj, tk)  = c(j, k, i)
            d10(tj, tk) = d1(j, k, i)
            d20(tj, tk) = d2(j, k, i)

            d10(tj, tk) = d10(tj, tk) - c0(tj, tk) * d11(tj, tk)
            d20(tj, tk) = d20(tj, tk) - c0(tj, tk) * d21(tj, tk)
            a0(tj, tk)  = a0(tj, tk) - c0(tj, tk) * a1(tj, tk)
            c0(tj, tk)  = -c0(tj, tk) * c1(tj, tk)

            a1(tj, tk)  = a0(tj, tk)
            c1(tj, tk)  = c0(tj, tk)
            d11(tj, tk) = d10(tj, tk)
            d21(tj, tk) = d20(tj, tk)

            a(j, k, i)  = a0(tj, tk)
            c(j, k, i)  = c0(tj, tk)
            d1(j, k, i) = d10(tj, tk)
            d2(j, k, i) = d20(tj, tk)
        enddo

        a0(tj, tk)  = a(j, k, 1)
        c0(tj, tk)  = c(j, k, 1)
        d10(tj, tk) = d1(j, k, 1)
        d20(tj, tk) = d2(j, k, 1)

        r0(tj, tk)  = 1.0d0 / (1.0d0 - a1(tj, tk) * c0(tj, tk))
        d10(tj, tk) = r0(tj, tk) * (d10(tj, tk) - c0(tj, tk) * d11(tj, tk))
        d20(tj, tk) = r0(tj, tk) * (d20(tj, tk) - c0(tj, tk) * d21(tj, tk))
        a0(tj, tk)  = r0(tj, tk) * a0(tj, tk)
        c0(tj, tk)  = -r0(tj, tk) * c0(tj, tk) * c1(tj, tk)

        d1(j, k, 1) = d10(tj, tk)
        d2(j, k, 1) = d20(tj, tk)
        a(j, k, 1)  = a0(tj, tk)
        c(j, k, 1)  = c0(tj, tk)

        a_rd(j, k, 1)  = a0(tj, tk)
        b_rd(j, k, 1)  = 1.0d0
        c_rd(j, k, 1)  = c0(tj, tk)
        d1_rd(j, k, 1) = d10(tj, tk)
        d2_rd(j, k, 1) = d20(tj, tk)

    end subroutine PaScaL_TDMA_many_modified_Thomas_2rhs_cuda

    !>
    !> @brief Pre-factor an ordinary Thomas system in-place.
    !> @details a stores normalized lower multipliers, b inverse pivots, and c
    !>          normalized upper coefficients.  No RHS is touched.
    !>
    attributes(global) subroutine tdma_static_prepare_cuda( &
        a, b, c, nx_sys, ny_sys, nz_row)

        implicit none

        integer, value, intent(in) :: nx_sys, ny_sys, nz_row
        double precision, device, intent(inout) :: a(:, :, :), b(:, :, :), c(:, :, :)
        integer :: i, j, k
        double precision :: inv_pivot, c_previous, lower

        j = (blockidx%x - 1) * blockdim%x + threadidx%x
        k = (blockidx%y - 1) * blockdim%y + threadidx%y
        if (j > nx_sys .or. k > ny_sys) return

        inv_pivot = 1.0d0 / b(j, k, 1)
        b(j, k, 1) = inv_pivot
        c(j, k, 1) = c(j, k, 1) * inv_pivot
        c_previous = c(j, k, 1)

        do i = 2, nz_row
            lower = a(j, k, i)
            inv_pivot = 1.0d0 / (b(j, k, i) - lower*c_previous)
            a(j, k, i) = lower * inv_pivot
            b(j, k, i) = inv_pivot
            c(j, k, i) = c(j, k, i) * inv_pivot
            c_previous = c(j, k, i)
        enddo

    end subroutine tdma_static_prepare_cuda


    !>
    !> @brief Apply precomputed ordinary Thomas factors to two RHS arrays.
    !>
    attributes(global) subroutine tdma_static_apply_2rhs_cuda( &
        a, b, c, d1, d2, nx_sys, ny_sys, nz_row)

        implicit none

        integer, value, intent(in) :: nx_sys, ny_sys, nz_row
        double precision, device, intent(in) :: a(:, :, :), b(:, :, :), c(:, :, :)
        double precision, device, intent(inout) :: d1(:, :, :), d2(:, :, :)
        integer :: i, j, k
        double precision :: previous1, previous2, current1, current2

        j = (blockidx%x - 1) * blockdim%x + threadidx%x
        k = (blockidx%y - 1) * blockdim%y + threadidx%y
        if (j > nx_sys .or. k > ny_sys) return

        previous1 = b(j, k, 1) * d1(j, k, 1)
        previous2 = b(j, k, 1) * d2(j, k, 1)
        d1(j, k, 1) = previous1
        d2(j, k, 1) = previous2

        do i = 2, nz_row
            current1 = b(j, k, i)*d1(j, k, i) &
                       - a(j, k, i)*previous1
            current2 = b(j, k, i)*d2(j, k, i) &
                       - a(j, k, i)*previous2
            d1(j, k, i) = current1
            d2(j, k, i) = current2
            previous1 = current1
            previous2 = current2
        enddo

        previous1 = d1(j, k, nz_row)
        previous2 = d2(j, k, nz_row)
        do i = nz_row-1, 1, -1
            current1 = d1(j, k, i) - c(j, k, i)*previous1
            current2 = d2(j, k, i) - c(j, k, i)*previous2
            d1(j, k, i) = current1
            d2(j, k, i) = current2
            previous1 = current1
            previous2 = current2
        enddo

    end subroutine tdma_static_apply_2rhs_cuda


    !>
    !> @brief Save the z-only lower/upper Poisson coefficients once.
    !>
    attributes(global) subroutine PaScaL_TDMA_copy_shared_rows_cuda( &
        a, c, lower_row, upper_row, sample_system, nz_row)

        implicit none

        integer, value, intent(in) :: sample_system, nz_row
        double precision, device, intent(in) :: a(:, :, :), c(:, :, :)
        double precision, device, intent(out) :: lower_row(:), upper_row(:)
        integer :: i

        if (blockidx%x /= 1 .or. threadidx%x /= 1) return
        do i = 1, nz_row
            ! System (1,1) is the pressure null-mode and has its first row
            ! replaced by the uniqueness constraint.  Prefer another spectral
            ! system when available so the shared physical z coefficient is
            ! retained; the null-mode correction itself is cached per system.
            lower_row(i) = a(sample_system, 1, i)
            upper_row(i) = c(sample_system, 1, i)
        enddo

    end subroutine PaScaL_TDMA_copy_shared_rows_cuda


    !>
    !> @brief Prepare coefficient-only modified Thomas factors.
    !> @details The final a/c response coefficients remain in a/c for the
    !>          solution update.  b stores inverse pivots.  Since the Poisson
    !>          lower/upper values are z-only, the RHS factors are reconstructed
    !>          from compact row vectors during apply.
    !>
    attributes(global) subroutine PaScaL_TDMA_many_prepare_modified_static_cuda( &
        a, b, c, row1_inverse, row1_upper, a_rd, b_rd, c_rd, &
        nx_sys, ny_sys, nz_row)

        implicit none

        integer, value, intent(in) :: nx_sys, ny_sys, nz_row
        double precision, device, intent(inout) :: a(:, :, :), b(:, :, :), c(:, :, :)
        double precision, device, intent(out) :: row1_inverse(:, :)
        double precision, device, intent(out) :: row1_upper(:, :)
        double precision, device, intent(out) :: a_rd(:, :, :), b_rd(:, :, :), c_rd(:, :, :)
        integer :: i, j, k
        double precision :: a0, a1, c0, c1, raw_a, inv_pivot, row1_factor

        j = (blockidx%x - 1) * blockdim%x + threadidx%x
        k = (blockidx%y - 1) * blockdim%y + threadidx%y
        if (j > nx_sys .or. k > ny_sys) return

        inv_pivot = 1.0d0 / b(j, k, 1)
        a0 = a(j, k, 1) * inv_pivot
        c0 = c(j, k, 1) * inv_pivot
        a(j, k, 1) = a0
        b(j, k, 1) = inv_pivot
        c(j, k, 1) = c0
        row1_upper(j, k) = c0

        inv_pivot = 1.0d0 / b(j, k, 2)
        a1 = a(j, k, 2) * inv_pivot
        c1 = c(j, k, 2) * inv_pivot
        a(j, k, 2) = a1
        b(j, k, 2) = inv_pivot
        c(j, k, 2) = c1

        do i = 3, nz_row
            a0 = a1
            c0 = c1
            raw_a = a(j, k, i)
            inv_pivot = 1.0d0 / (b(j, k, i) - raw_a*c0)
            a1 = -(raw_a*inv_pivot)*a0
            c1 = c(j, k, i)*inv_pivot
            a(j, k, i) = a1
            b(j, k, i) = inv_pivot
            c(j, k, i) = c1
        enddo

        a_rd(j, k, 2) = a1
        b_rd(j, k, 2) = 1.0d0
        c_rd(j, k, 2) = c1

        ! a0/c0 are the forward coefficients at row nz_row-1.
        a1 = a0
        c1 = c0
        do i = nz_row-2, 2, -1
            a0 = a(j, k, i) - c(j, k, i)*a1
            c0 = -c(j, k, i)*c1
            a(j, k, i) = a0
            c(j, k, i) = c0
            a1 = a0
            c1 = c0
        enddo

        a0 = a(j, k, 1)
        c0 = c(j, k, 1)
        row1_factor = 1.0d0 / (1.0d0 - a1*c0)
        row1_inverse(j, k) = row1_factor
        a0 = row1_factor*a0
        c0 = -row1_factor*c0*c1
        a(j, k, 1) = a0
        c(j, k, 1) = c0

        a_rd(j, k, 1) = a0
        b_rd(j, k, 1) = 1.0d0
        c_rd(j, k, 1) = c0

    end subroutine PaScaL_TDMA_many_prepare_modified_static_cuda


    !>
    !> @brief Apply cached modified-Thomas factors to two new local RHS arrays.
    !>
    attributes(global) subroutine PaScaL_TDMA_many_apply_modified_static_2rhs_cuda( &
        inverse_pivot, lower_row, upper_row, row1_inverse, row1_upper, &
        d1, d2, d1_rd, d2_rd, &
        nx_sys, ny_sys, nz_row)

        implicit none

        integer, value, intent(in) :: nx_sys, ny_sys, nz_row
        double precision, device, intent(in) :: inverse_pivot(:, :, :)
        double precision, device, intent(in) :: lower_row(:), upper_row(:)
        double precision, device, intent(in) :: row1_inverse(:, :)
        double precision, device, intent(in) :: row1_upper(:, :)
        double precision, device, intent(inout) :: d1(:, :, :), d2(:, :, :)
        double precision, device, intent(out) :: d1_rd(:, :, :), d2_rd(:, :, :)
        integer :: i, j, k
        double precision :: previous1, previous2, current1, current2

        j = (blockidx%x - 1) * blockdim%x + threadidx%x
        k = (blockidx%y - 1) * blockdim%y + threadidx%y
        if (j > nx_sys .or. k > ny_sys) return

        d1(j, k, 1) = inverse_pivot(j, k, 1)*d1(j, k, 1)
        d2(j, k, 1) = inverse_pivot(j, k, 1)*d2(j, k, 1)
        d1(j, k, 2) = inverse_pivot(j, k, 2)*d1(j, k, 2)
        d2(j, k, 2) = inverse_pivot(j, k, 2)*d2(j, k, 2)

        previous1 = d1(j, k, 2)
        previous2 = d2(j, k, 2)
        do i = 3, nz_row
            current1 = inverse_pivot(j, k, i)*d1(j, k, i) &
                       - lower_row(i)*inverse_pivot(j, k, i)*previous1
            current2 = inverse_pivot(j, k, i)*d2(j, k, i) &
                       - lower_row(i)*inverse_pivot(j, k, i)*previous2
            d1(j, k, i) = current1
            d2(j, k, i) = current2
            previous1 = current1
            previous2 = current2
        enddo

        d1_rd(j, k, 2) = d1(j, k, nz_row)
        d2_rd(j, k, 2) = d2(j, k, nz_row)

        do i = nz_row-2, 2, -1
            d1(j, k, i) = d1(j, k, i) &
                          - upper_row(i)*inverse_pivot(j, k, i)*d1(j, k, i+1)
            d2(j, k, i) = d2(j, k, i) &
                          - upper_row(i)*inverse_pivot(j, k, i)*d2(j, k, i+1)
        enddo

        d1(j, k, 1) = row1_inverse(j, k) &
                       * (d1(j, k, 1) &
                          - row1_upper(j, k)*d1(j, k, 2))
        d2(j, k, 1) = row1_inverse(j, k) &
                       * (d2(j, k, 1) &
                          - row1_upper(j, k)*d2(j, k, 2))
        d1_rd(j, k, 1) = d1(j, k, 1)
        d2_rd(j, k, 1) = d2(j, k, 1)

    end subroutine PaScaL_TDMA_many_apply_modified_static_2rhs_cuda

    !>
    !> @brief   The modified Thomas algorithm : elimination of lower diagonal elements
    !> @param   a           Coefficient array of lower diagonal elements
    !> @param   c           Coefficient array of upper diagonal elements
    !> @param   d           Coefficient array of solution terms
    !> @param   d_rd        Reduced coefficient array of solution terms
    !> @param   nz_row      Row size of partitioned tridiagonal matrix in z-direction per process
    !>
    attributes(global) subroutine PaScaL_TDMA_many_update_solution_cuda(a, c, d, d_rd, nx_sys, ny_sys, nz_row)

        implicit none

        integer, value, intent(in)      :: nx_sys, ny_sys, nz_row
        double precision, device, intent(in)    :: a(:, :, :), c(:, :, :), d_rd(:, :, :)
        double precision, device, intent(inout) :: d(:, :, :)

        ! Temporary variables for computation
        integer :: i, j, k
        integer :: tj, tk

        ! Block shared memory
        double precision, shared :: ds(blockdim%x + 1, blockdim%y), de(blockdim%x + 1, blockdim%y)

        ! Global index
        j = (blockidx%x - 1) * blockdim%x + threadidx%x
        k = (blockidx%y - 1) * blockdim%y + threadidx%y

        ! Local index in block
        tj = threadidx%x
        tk = threadidx%y

        ! The final CUDA block can be partial after ceiling division.
        ! ds/de are private to each (tj,tk), so the original syncthreads() was unnecessary.
        if (j > nx_sys .or. k > ny_sys) return

        ! Using shared memory
		! First and second indices are for thread IDs
        ds(tj, tk) = d_rd(j, k, 1)
        de(tj, tk) = d_rd(j, k, 2)

        ! Update solutions of the modified tridiagonal system with the solutions of the reduced tridiagonal system.
        d(j, k, 1)      = ds(tj, tk)
        d(j, k, nz_row) = de(tj, tk)

        do i = 2, nz_row-1
            d(j, k, i) = d(j, k, i) - a(j, k, i) * ds(tj, tk) - c(j, k, i) * de(tj, tk)
        enddo

    end subroutine PaScaL_TDMA_many_update_solution_cuda

    !>
    !> @brief Update two full solutions from their reduced boundary values.
    !>
    attributes(global) subroutine PaScaL_TDMA_many_update_solution_2rhs_cuda( &
        a, c, d1, d2, d1_rd, d2_rd, nx_sys, ny_sys, nz_row)

        implicit none

        integer, value, intent(in) :: nx_sys, ny_sys, nz_row
        double precision, device, intent(in) :: a(:, :, :), c(:, :, :)
        double precision, device, intent(in) :: d1_rd(:, :, :), d2_rd(:, :, :)
        double precision, device, intent(inout) :: d1(:, :, :), d2(:, :, :)

        integer :: i, j, k
        integer :: tj, tk
        double precision, shared :: d1s(blockdim%x + 1, blockdim%y)
        double precision, shared :: d1e(blockdim%x + 1, blockdim%y)
        double precision, shared :: d2s(blockdim%x + 1, blockdim%y)
        double precision, shared :: d2e(blockdim%x + 1, blockdim%y)

        j = (blockidx%x - 1) * blockdim%x + threadidx%x
        k = (blockidx%y - 1) * blockdim%y + threadidx%y
        tj = threadidx%x
        tk = threadidx%y

        if (j > nx_sys .or. k > ny_sys) return

        d1s(tj, tk) = d1_rd(j, k, 1)
        d1e(tj, tk) = d1_rd(j, k, 2)
        d2s(tj, tk) = d2_rd(j, k, 1)
        d2e(tj, tk) = d2_rd(j, k, 2)

        d1(j, k, 1)      = d1s(tj, tk)
        d1(j, k, nz_row) = d1e(tj, tk)
        d2(j, k, 1)      = d2s(tj, tk)
        d2(j, k, nz_row) = d2e(tj, tk)

        do i = 2, nz_row-1
            d1(j, k, i) = d1(j, k, i) &
                          - a(j, k, i) * d1s(tj, tk) &
                          - c(j, k, i) * d1e(tj, tk)
            d2(j, k, i) = d2(j, k, i) &
                          - a(j, k, i) * d2s(tj, tk) &
                          - c(j, k, i) * d2e(tj, tk)
        enddo

    end subroutine PaScaL_TDMA_many_update_solution_2rhs_cuda


    !>
    !> @brief   Subroutine to transpose x-y slab to y-z slab for solving TDM in z-direction
    !> @param   slab_xy     Coefficient array in the shape of x-y slab
    !> @param   slab_yz     Coefficient array in the shape of y-z slab
    !>
    subroutine transpose_slab_xy_to_yz (p, slab_xy, slab_yz)

        implicit none

        type(ptdma_plan_many_cuda), intent(inout)   :: p
        double precision, device, intent(in )       :: slab_xy(:, :, :)
        double precision, device, intent(out)       :: slab_yz(:, :, :)
        
        integer  :: nblksize
        integer  :: ierr
        double precision :: prof_t0
        integer :: profile_base

        nblksize = p%nx_sys * p%ny_sys * p%nz_row_rd / p%nprocs
        ptdma_xy_call = ptdma_xy_call + 1
        profile_base = 3 + 3 * (ptdma_xy_call - 1)

        call ptdma_profile_start(prof_t0)
        call mem_detach_slab_xy <<<p%blocks_alltoall, p%threads>>> &
                                (slab_xy, p%sendbuf, p%nx_sys, p%ny_sys, p%nz_row_rd, p%nprocs)
        if (profile_base >= 3 .and. profile_base <= 12) then
            call ptdma_profile_stop( &
                prof_t0, ptdma_profile_times(profile_base))
        endif

        !----- alltoall communication of sbuf to rbuf
        call ptdma_profile_start(prof_t0)
        ierr = cudaStreamSynchronize()
        call MPI_Alltoall(p%sendbuf, nblksize, MPI_DOUBLE_PRECISION, &
                          p%recvbuf, nblksize, MPI_DOUBLE_PRECISION, &
                          p%ptdma_world, ierr)
        if (profile_base >= 3 .and. profile_base <= 12) then
            call ptdma_profile_stop( &
                prof_t0, ptdma_profile_times(profile_base + 1))
        endif

        call ptdma_profile_start(prof_t0)
        call mem_unite_slab_yz<<<p%blocks_alltoall, p%threads>>> &
                                (p%recvbuf, slab_yz, p%nx_sys, p%ny_sys, p%nz_row_rd, p%nprocs)
        if (profile_base >= 3 .and. profile_base <= 12) then
            call ptdma_profile_stop( &
                prof_t0, ptdma_profile_times(profile_base + 2))
        endif
  
    end subroutine transpose_slab_xy_to_yz

    !>
    !> @brief   Subroutine to rearrange x-y slab to a 1D array for MPI_Alltoall
    !> @param   slab_xy     Coefficient array in the shape of x-y slab
    !> @param   array1D     1-D Coefficient array
    !> @param   n1, n2, n3  Dimension of array 'slab_xy'
    !> @param   nprocs      Number of processes in 'mpi_comm'
    !>
    attributes(global) subroutine mem_detach_slab_xy(slab_xy, array1D, n1, n2, n3, nprocs)

        implicit none

        integer, value, intent(in)  :: n1, n2, n3, nprocs
        double precision, device, intent(in)    :: slab_xy(:, :, :)
        double precision, device, intent(out)   :: array1D(:)

        ! Variables to calculate indices for in and out arrays
        integer :: i, j, k, kblk, n1blksize, blksize
        integer :: pos, pos_i, pos_j, pos_k

        n1blksize  = n1 / nprocs
        blksize    = n1blksize * n2 * n3

        i = (blockidx%x - 1) * blockdim%x + threadidx%x
        j = (blockidx%y - 1) * blockdim%y + threadidx%y
        k = (blockidx%z - 1) * blockdim%z + threadidx%z

        if (i > n1blksize .or. j > n2 .or. k > n3) return

        pos_i = (i - 1) * n2 * n3
        pos_j = (j - 1) * n3

        do kblk = 1, nprocs
            pos_k = k + (kblk - 1) * blksize
            pos   = pos_k + pos_j + pos_i
            array1D(pos) = slab_xy(i + (kblk - 1) * n1blksize, j, k)
        enddo

    end subroutine mem_detach_slab_xy

    !>
    !> @brief   Subroutine to rearrange a 1D array to y-z slab after MPI_Alltoall
    !> @param   array1D     1-D Coefficient array
    !> @param   slab_yz     Coefficient array in the shape of y-z slab
    !> @param   n1, n2, n3  Dimension of array 'slab_xy'
    !> @param   nprocs      Number of processes in 'mpi_comm'
    !>
    attributes(global) subroutine mem_unite_slab_yz(array1D, slab_yz, n1, n2, n3, nprocs)

        implicit none

        integer, value, intent(in)  :: n1, n2, n3, nprocs
        double precision, device, intent(in)    :: array1D(:)
        double precision, device, intent(out)   :: slab_yz(:, :, :)

        ! Variables to calculate indices for in and out arrays
        integer :: i, j, k, kblk, blksize
        integer :: pos, pos_i, pos_j, pos_k

        blksize  = n1 * n2 * n3 / nprocs

        i = (blockidx%x - 1) * blockdim%x + threadidx%x
        j = (blockidx%y - 1) * blockdim%y + threadidx%y
        k = (blockidx%z - 1) * blockdim%z + threadidx%z

        if (i > n1 / nprocs .or. j > n2 .or. k > n3) return

        pos_i = (i - 1) * n2 * n3
        pos_j = (j - 1) * n3

        do kblk = 1, nprocs
            pos_k = k + (kblk - 1) * blksize
            pos   = pos_k + pos_j + pos_i
            slab_yz(i, j, k + (kblk - 1) * n3) = array1D(pos)
        enddo

    end subroutine mem_unite_slab_yz

    !>
    !> @brief   Subroutine to transpose y-z slab (d_rt_d) to x-y slab (d_rd_d) after solving TDM in z-direction
    !> @param   p           Plan for a single tridiagonal system of equations
    !> @param   slab_yz     Coefficient array in the shape of y-z slab
    !> @param   slab_xy     Coefficient array in the shape of x-y slab
    !>
    subroutine transpose_slab_yz_to_xy(p, slab_yz, slab_xy)

        implicit none

        type(ptdma_plan_many_cuda), intent(inout)      :: p
        double precision, device, intent(in )   :: slab_yz(:, :, :)
        double precision, device, intent(out)   :: slab_xy(:, :, :)

        integer  :: nblksize
        integer  :: ierr
        double precision :: prof_t0
  
        nblksize   = p%nx_sys * p%ny_sys * p%nz_row_rd / p%nprocs

        call ptdma_profile_start(prof_t0)
        call mem_detach_slab_yz<<<p%blocks_alltoall, p%threads>>> &
                                (slab_yz, p%sendbuf, p%nx_sys, p%ny_sys, p%nz_row_rd, p%nprocs)
        call ptdma_profile_stop( &
            prof_t0, ptdma_profile_times(16))

        !----- alltoall communication of sendbuf to recvbuf
        call ptdma_profile_start(prof_t0)
        ierr = cudaStreamSynchronize()
        call MPI_Alltoall(p%sendbuf, nblksize, MPI_DOUBLE_PRECISION, &
                          p%recvbuf, nblksize, MPI_DOUBLE_PRECISION, &
                          p%ptdma_world, ierr)
        call ptdma_profile_stop( &
            prof_t0, ptdma_profile_times(17))

        call ptdma_profile_start(prof_t0)
        call mem_unite_slab_xy<<<p%blocks_alltoall, p%threads>>> &
                                (p%recvbuf, slab_xy, p%nx_sys, p%ny_sys, p%nz_row_rd, p%nprocs)
        call ptdma_profile_stop( &
            prof_t0, ptdma_profile_times(18))

    end subroutine transpose_slab_yz_to_xy

    !>
    !> @brief   Subroutine to rearrange y-z slab to a 1D array for MPI_Alltoall
    !> @param   slab_yz     Coefficient array in the shape of y-z slab
    !> @param   array1D     1-D Coefficient array
    !> @param   n1, n2, n3  Dimension of array 'slab_xy'
    !> @param   nprocs      Number of processes in 'mpi_comm'
    !>
    attributes(global) subroutine mem_detach_slab_yz(slab_yz, array1D, n1, n2, n3, nprocs)

        implicit none

        integer, value, intent(in)  :: n1, n2, n3, nprocs
        double precision, device, intent(in)    :: slab_yz(:, :, :)
        double precision, device, intent(out)   :: array1D(:)

        ! Variables to calculate indices for in and out arrays
        integer :: i, j, k, kblk, blksize
        integer :: pos_k, pos_i, pos_j, pos

        i = (blockidx%x - 1) * blockdim%x + threadidx%x
        j = (blockidx%y - 1) * blockdim%y + threadidx%y
        k = (blockidx%z - 1) * blockdim%z + threadidx%z

        if (i > n1 / nprocs .or. j > n2 .or. k > n3) return

        pos_i = (i - 1) * n2 * n3
        pos_j = (j - 1) * n3
        blksize  = n1 * n2 * n3 / nprocs

        do kblk = 1, nprocs
            pos_k = k + (kblk - 1) * blksize
            pos   = pos_k + pos_j + pos_i
            array1D(pos) = slab_yz(i, j, k + (kblk - 1) * n3)
        enddo

    end subroutine mem_detach_slab_yz

    !>
    !> @brief   Subroutine to rearrange a 1D array to x-y slab after MPI_Alltoall
    !> @param   array1D     1-D Coefficient array
    !> @param   slab_xy     Coefficient array in the shape of x-y slab
    !> @param   n1, n2, n3  Dimension of array 'slab_xy'
    !> @param   nprocs      Number of processes in 'mpi_comm'
    !>
    attributes(global) subroutine mem_unite_slab_xy(array1D, slab_xy, n1, n2, n3, nprocs)

        implicit none

        integer, value, intent(in)  :: n1, n2, n3, nprocs
        double precision, device, intent(in)    :: array1D(:)
        double precision, device, intent(out)   :: slab_xy(:, :, :)

        ! Variables to calculate indices for in and out arrays
        integer :: i, j, k, kblk, n1blksize, blksize
        integer :: pos, pos_i, pos_j, pos_k

        n1blksize  = n1 / nprocs
        blksize  = n1blksize * n2 * n3

        i = (blockidx%x - 1) * blockdim%x + threadidx%x
        j = (blockidx%y - 1) * blockdim%y + threadidx%y
        k = (blockidx%z - 1) * blockdim%z + threadidx%z

        if (i > n1blksize .or. j > n2 .or. k > n3) return

        pos_i = (i - 1) * n2 * n3
        pos_j = (j - 1) * n3

        do kblk = 1, nprocs
            pos_k = k + (kblk - 1) * blksize
            pos   = pos_k + pos_j + pos_i
            slab_xy(i + (kblk - 1) * n1blksize, j, k) = array1D(pos)
        enddo

    end subroutine mem_unite_slab_xy

    !>
    !> @brief Transpose two RHS slabs with one MPI_Alltoall.
    !>
    subroutine transpose_slab_xy_to_yz_2rhs(p, xy1, xy2, yz1, yz2)

        implicit none

        type(ptdma_plan_many_cuda), intent(inout) :: p
        double precision, device, intent(in)  :: xy1(:, :, :), xy2(:, :, :)
        double precision, device, intent(out) :: yz1(:, :, :), yz2(:, :, :)
        integer :: nblksize, ierr, profile_base
        double precision :: prof_t0

        nblksize = p%nx_sys * p%ny_sys * p%nz_row_rd / p%nprocs
        ptdma_xy_call = ptdma_xy_call + 1
        ! A/B/C occupy phases 3:11; the packed RHS pair is phase 12:14.
        profile_base = 12

        call ptdma_profile_start(prof_t0)
        call mem_detach_slab_xy_2rhs<<<p%blocks_alltoall, p%threads>>> &
            (xy1, xy2, p%sendbuf_2rhs, &
             p%nx_sys, p%ny_sys, p%nz_row_rd, p%nprocs)
        call ptdma_profile_stop(prof_t0, ptdma_profile_times(profile_base))

        call ptdma_profile_start(prof_t0)
        ierr = cudaStreamSynchronize()
        call MPI_Alltoall(p%sendbuf_2rhs, 2*nblksize, MPI_DOUBLE_PRECISION, &
                          p%recvbuf_2rhs, 2*nblksize, MPI_DOUBLE_PRECISION, &
                          p%ptdma_world, ierr)
        call ptdma_profile_stop(prof_t0, ptdma_profile_times(profile_base + 1))

        call ptdma_profile_start(prof_t0)
        call mem_unite_slab_yz_2rhs<<<p%blocks_alltoall, p%threads>>> &
            (p%recvbuf_2rhs, yz1, yz2, &
             p%nx_sys, p%ny_sys, p%nz_row_rd, p%nprocs)
        call ptdma_profile_stop(prof_t0, ptdma_profile_times(profile_base + 2))

    end subroutine transpose_slab_xy_to_yz_2rhs

    !>
    !> @brief Transpose two solution slabs back with one MPI_Alltoall.
    !>
    subroutine transpose_slab_yz_to_xy_2rhs(p, yz1, yz2, xy1, xy2)

        implicit none

        type(ptdma_plan_many_cuda), intent(inout) :: p
        double precision, device, intent(in)  :: yz1(:, :, :), yz2(:, :, :)
        double precision, device, intent(out) :: xy1(:, :, :), xy2(:, :, :)
        integer :: nblksize, ierr
        double precision :: prof_t0

        nblksize = p%nx_sys * p%ny_sys * p%nz_row_rd / p%nprocs

        call ptdma_profile_start(prof_t0)
        call mem_detach_slab_yz_2rhs<<<p%blocks_alltoall, p%threads>>> &
            (yz1, yz2, p%sendbuf_2rhs, &
             p%nx_sys, p%ny_sys, p%nz_row_rd, p%nprocs)
        call ptdma_profile_stop(prof_t0, ptdma_profile_times(16))

        call ptdma_profile_start(prof_t0)
        ierr = cudaStreamSynchronize()
        call MPI_Alltoall(p%sendbuf_2rhs, 2*nblksize, MPI_DOUBLE_PRECISION, &
                          p%recvbuf_2rhs, 2*nblksize, MPI_DOUBLE_PRECISION, &
                          p%ptdma_world, ierr)
        call ptdma_profile_stop(prof_t0, ptdma_profile_times(17))

        call ptdma_profile_start(prof_t0)
        call mem_unite_slab_xy_2rhs<<<p%blocks_alltoall, p%threads>>> &
            (p%recvbuf_2rhs, xy1, xy2, &
             p%nx_sys, p%ny_sys, p%nz_row_rd, p%nprocs)
        call ptdma_profile_stop(prof_t0, ptdma_profile_times(18))

    end subroutine transpose_slab_yz_to_xy_2rhs

    attributes(global) subroutine mem_detach_slab_xy_2rhs( &
        slab1, slab2, array1d, n1, n2, n3, nprocs)

        implicit none

        integer, value, intent(in) :: n1, n2, n3, nprocs
        double precision, device, intent(in)  :: slab1(:, :, :), slab2(:, :, :)
        double precision, device, intent(out) :: array1d(:)
        integer :: i, j, k, peer, n1blk, blksize, local_pos, peer_base

        n1blk = n1 / nprocs
        blksize = n1blk * n2 * n3
        i = (blockidx%x - 1) * blockdim%x + threadidx%x
        j = (blockidx%y - 1) * blockdim%y + threadidx%y
        k = (blockidx%z - 1) * blockdim%z + threadidx%z
        if (i > n1blk .or. j > n2 .or. k > n3) return

        local_pos = (i - 1) * n2 * n3 + (j - 1) * n3 + k
        do peer = 1, nprocs
            peer_base = (peer - 1) * 2 * blksize
            array1d(peer_base + local_pos) = &
                slab1(i + (peer - 1) * n1blk, j, k)
            array1d(peer_base + blksize + local_pos) = &
                slab2(i + (peer - 1) * n1blk, j, k)
        enddo

    end subroutine mem_detach_slab_xy_2rhs

    attributes(global) subroutine mem_unite_slab_yz_2rhs( &
        array1d, slab1, slab2, n1, n2, n3, nprocs)

        implicit none

        integer, value, intent(in) :: n1, n2, n3, nprocs
        double precision, device, intent(in)  :: array1d(:)
        double precision, device, intent(out) :: slab1(:, :, :), slab2(:, :, :)
        integer :: i, j, k, peer, n1blk, blksize, local_pos, peer_base

        n1blk = n1 / nprocs
        blksize = n1blk * n2 * n3
        i = (blockidx%x - 1) * blockdim%x + threadidx%x
        j = (blockidx%y - 1) * blockdim%y + threadidx%y
        k = (blockidx%z - 1) * blockdim%z + threadidx%z
        if (i > n1blk .or. j > n2 .or. k > n3) return

        local_pos = (i - 1) * n2 * n3 + (j - 1) * n3 + k
        do peer = 1, nprocs
            peer_base = (peer - 1) * 2 * blksize
            slab1(i, j, k + (peer - 1) * n3) = &
                array1d(peer_base + local_pos)
            slab2(i, j, k + (peer - 1) * n3) = &
                array1d(peer_base + blksize + local_pos)
        enddo

    end subroutine mem_unite_slab_yz_2rhs

    attributes(global) subroutine mem_detach_slab_yz_2rhs( &
        slab1, slab2, array1d, n1, n2, n3, nprocs)

        implicit none

        integer, value, intent(in) :: n1, n2, n3, nprocs
        double precision, device, intent(in)  :: slab1(:, :, :), slab2(:, :, :)
        double precision, device, intent(out) :: array1d(:)
        integer :: i, j, k, peer, n1blk, blksize, local_pos, peer_base

        n1blk = n1 / nprocs
        blksize = n1blk * n2 * n3
        i = (blockidx%x - 1) * blockdim%x + threadidx%x
        j = (blockidx%y - 1) * blockdim%y + threadidx%y
        k = (blockidx%z - 1) * blockdim%z + threadidx%z
        if (i > n1blk .or. j > n2 .or. k > n3) return

        local_pos = (i - 1) * n2 * n3 + (j - 1) * n3 + k
        do peer = 1, nprocs
            peer_base = (peer - 1) * 2 * blksize
            array1d(peer_base + local_pos) = &
                slab1(i, j, k + (peer - 1) * n3)
            array1d(peer_base + blksize + local_pos) = &
                slab2(i, j, k + (peer - 1) * n3)
        enddo

    end subroutine mem_detach_slab_yz_2rhs

    attributes(global) subroutine mem_unite_slab_xy_2rhs( &
        array1d, slab1, slab2, n1, n2, n3, nprocs)

        implicit none

        integer, value, intent(in) :: n1, n2, n3, nprocs
        double precision, device, intent(in)  :: array1d(:)
        double precision, device, intent(out) :: slab1(:, :, :), slab2(:, :, :)
        integer :: i, j, k, peer, n1blk, blksize, local_pos, peer_base

        n1blk = n1 / nprocs
        blksize = n1blk * n2 * n3
        i = (blockidx%x - 1) * blockdim%x + threadidx%x
        j = (blockidx%y - 1) * blockdim%y + threadidx%y
        k = (blockidx%z - 1) * blockdim%z + threadidx%z
        if (i > n1blk .or. j > n2 .or. k > n3) return

        local_pos = (i - 1) * n2 * n3 + (j - 1) * n3 + k
        do peer = 1, nprocs
            peer_base = (peer - 1) * 2 * blksize
            slab1(i + (peer - 1) * n1blk, j, k) = &
                array1d(peer_base + local_pos)
            slab2(i + (peer - 1) * n1blk, j, k) = &
                array1d(peer_base + blksize + local_pos)
        enddo

    end subroutine mem_unite_slab_xy_2rhs

end module PaScaL_TDMA_cuda
