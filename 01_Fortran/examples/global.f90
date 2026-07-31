module global
    use mpi
    implicit none
    
#ifdef SINGLE_PRECISION
    integer, parameter :: rp = kind(0.0), MPI_real_type = MPI_REAL, MPI_complex_type = MPI_COMPLEX
#else
    integer, parameter :: rp = kind(0.0d0), MPI_real_type = MPI_DOUBLE_PRECISION, MPI_complex_type = MPI_DOUBLE_COMPLEX
#endif
    real(rp), parameter :: PI = real(dacos(-1.0d0),rp)

    ! 1 - Domain Settings:
    ! 1-1. Size and Volume
    real(rp) :: x1_start, x2_start, x3_start
    real(rp) :: x1_end  , x2_end  , x3_end
    real(rp) :: L1, L2, L3
    real(rp) :: Volume

    ! 1-2. Mesh
    integer  ::  n1,  n2,  n3
    integer  :: n1m, n2m, n3m
    integer  :: n1p, n2p, n3p
    real(rp) :: resolution_x, resolution_y, resolution_z
    
    ! 2. Mesh Settings:
    ! 2-1. Grid Style
    logical  :: pbc1, pbc2, pbc3
    integer  :: uniform1, uniform2, uniform3
    real(rp) :: gamma1, gamma2, gamma3
    
    ! 3. Parallel Settings:
    ! 3-1. MPI Procs
    integer  :: np1, np2, np3
    ! 3-2. CUDA Thread
    integer  :: thread_in_x, thread_in_y, thread_in_z
    ! 3-3. CUDA Pascal TDMA Thread
    integer  :: thread_in_x_pascal, thread_in_y_pascal, thread_in_fft_pascal
    
    integer  :: Poisson_model

    ! 7. Post Options:
    logical  :: ContinueFilein, ContinueFileout
    
    ! 8. Directories:
    character(len=128) :: dir_cont_filein, dir_cont_fileout, dir_instantfield

    !integer  :: print_j_index_wall,print_j_index_bulk,print_k_index

    contains

    subroutine global_inputpara(myrank)
        implicit none

        integer, intent(in) :: myrank

        integer :: mpi_size
        integer :: ierr
        integer :: input_unit
        integer :: i
        
        logical :: file_exists

        character(len=256) :: input_basename
        character(len=256) :: input_filename
        character(len=256) :: candidates(3)

        ! Namelist variables for file input
        namelist /domain_size/ L1, L2, L3

        namelist /mesh/ n1m, n2m, n3m

        namelist /grid_style/ pbc1, pbc2, pbc3, &
                              uniform1, uniform2, uniform3, &
                              gamma1, gamma2, gamma3
        
        namelist /MPI_procs/ np1, np2, np3

        namelist /cuda_thread/ thread_in_x, &
                               thread_in_y, &
                               thread_in_z
        
        namelist /cuda_pascal_tdma_thread/ &
            thread_in_x_pascal, &
            thread_in_y_pascal, &
            thread_in_fft_pascal

        namelist /Poisson/ Poisson_model

        namelist /post_option/ ContinueFilein, ContinueFileout

        namelist /directories/ dir_cont_filein, &
                               dir_cont_fileout, &
                               dir_instantfield
        
        call MPI_Comm_size(MPI_COMM_WORLD, mpi_size, ierr)

        if (ierr /= MPI_SUCCESS) then
            if (myrank == 0) then
                write(*,*) '[ERROR] MPI_Comm_size failed'
            endif

            call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
        endif

        write(input_basename, '("PARA_INPUT_",I0,".dat")') mpi_size

        ! 1. Executed inside run/
        ! 2. Executed from project root
        ! 3. Executed inside examples/
        candidates(1) = trim(input_basename)
        candidates(2) = 'run/'    // trim(input_basename)
        candidates(3) = '../run/' // trim(input_basename)

        input_filename = ''
        file_exists = .false.

        do i = 1, size(candidates)

            inquire( &
                 file  = trim(candidates(i)), &
                 exist = file_exists          &
            )

            if (file_exists) then
                input_filename = trim(candidates(i))
                exit
            endif
        enddo
        ! ============================================================
        ! Abort if the corresponding file does not exist
        ! ============================================================
        if (.not. file_exists) then

            if (myrank == 0) then
                write(*,*)
                write(*,*) '============================================================'
                write(*,*) '[ERROR] MPI-dependent input file was not found.'
                write(*,*) 'MPI size       : ', mpi_size
                write(*,*) 'Expected file  : ', trim(input_basename)
                write(*,*) 'Searched paths :'
                write(*,*) '  ', trim(candidates(1))
                write(*,*) '  ', trim(candidates(2))
                write(*,*) '  ', trim(candidates(3))
                write(*,*) '============================================================'
                write(*,*)
            endif

            call MPI_Abort(MPI_COMM_WORLD, 1, ierr)

        endif


        if (myrank == 0) then
            write(*,*)
            write(*,*) '============================================================'
            write(*,*) '[INPUT] MPI size      : ', mpi_size
            write(*,*) '[INPUT] Parameter file: ', trim(input_filename)
            write(*,*) '============================================================'
            write(*,*)
        endif


        ! ============================================================
        ! Read rank-dependent parameter file
        ! ============================================================
        open( &
            newunit = input_unit,             &
            file    = trim(input_filename),   &
            status  = 'old',                  &
            action  = 'read',                 &
            iostat  = ierr                    &
        )

        if (ierr /= 0) then

            if (myrank == 0) then
                write(*,*) '[ERROR] Unable to open input file:'
                write(*,*) trim(input_filename)
                write(*,*) 'IOSTAT = ', ierr
            endif

            call MPI_Abort(MPI_COMM_WORLD, 1, ierr)

        endif


        read(input_unit, nml=domain_size, iostat=ierr)
        if (ierr /= 0) then
            if (myrank == 0) write(*,*) '[ERROR] Failed to read /domain_size/.'
            call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
        endif

        read(input_unit, nml=mesh, iostat=ierr)
        if (ierr /= 0) then
            if (myrank == 0) write(*,*) '[ERROR] Failed to read /mesh/.'
            call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
        endif

        read(input_unit, nml=grid_style, iostat=ierr)
        if (ierr /= 0) then
            if (myrank == 0) write(*,*) '[ERROR] Failed to read /grid_style/.'
            call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
        endif

        read(input_unit, nml=MPI_procs, iostat=ierr)
        if (ierr /= 0) then
            if (myrank == 0) write(*,*) '[ERROR] Failed to read /MPI_procs/.'
            call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
        endif

        read(input_unit, nml=cuda_thread, iostat=ierr)
        if (ierr /= 0) then
            if (myrank == 0) write(*,*) '[ERROR] Failed to read /cuda_thread/.'
            call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
        endif

        read(input_unit, nml=cuda_pascal_tdma_thread, iostat=ierr)
        if (ierr /= 0) then
            if (myrank == 0) then
                write(*,*) '[ERROR] Failed to read /cuda_pascal_tdma_thread/.'
            endif
            call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
        endif

        read(input_unit, nml=Poisson, iostat=ierr)
        if (ierr /= 0) then
            if (myrank == 0) write(*,*) '[ERROR] Failed to read /Poisson/.'
            call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
        endif

        read(input_unit, nml=post_option, iostat=ierr)
        if (ierr /= 0) then
            if (myrank == 0) write(*,*) '[ERROR] Failed to read /post_option/.'
            call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
        endif

        read(input_unit, nml=directories, iostat=ierr)
        if (ierr /= 0) then
            if (myrank == 0) write(*,*) '[ERROR] Failed to read /directories/.'
            call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
        endif

        close(input_unit)


        ! ============================================================
        ! Validate MPI decomposition
        ! ============================================================
        if (np1 * np2 * np3 /= mpi_size) then

            if (myrank == 0) then
                write(*,*)
                write(*,*) '============================================================'
                write(*,*) '[ERROR] MPI decomposition mismatch.'
                write(*,*) 'MPI ranks from launcher : ', mpi_size
                write(*,*) 'np1, np2, np3           : ', np1, np2, np3
                write(*,*) 'np1*np2*np3             : ', np1*np2*np3
                write(*,*) 'Input file              : ', trim(input_filename)
                write(*,*) '============================================================'
                write(*,*)
            endif

            call MPI_Abort(MPI_COMM_WORLD, 1, ierr)

        endif


        ! ============================================================
        ! Domain initialization
        ! ============================================================
        x1_start = real(0.0, rp)
        x2_start = real(0.0, rp)
        x3_start = real(0.0, rp)

        x1_end = x1_start + L1
        x2_end = x2_start + L2
        x3_end = x3_start + L3

        Volume = L1 * L2 * L3


        ! ============================================================
        ! Mesh initialization
        ! ============================================================
        n1  = n1m + 1
        n1p = n1  + 1

        n2  = n2m + 1
        n2p = n2  + 1

        n3  = n3m + 1
        n3p = n3  + 1

        resolution_x = real(n1m, rp) / L1
        resolution_y = real(n2m, rp) / L2
        resolution_z = real(n3m, rp) / L3


        call global_print_settings(myrank)

    end subroutine global_inputpara

    subroutine global_print_settings(myrank)
        implicit none
        integer, intent(in) :: myrank
        integer             :: ierr
    
        if (myrank == 0) then

            write(*, '(A)') '================== Poisson solver with Fourier Transform =================='
            write(*, '(A)') '1 - Domain Settings:'
            write(*, '(A, 3F12.2)') '  - Start (x1, x2, x3):', x1_start, x2_start, x3_start
            write(*, '(A, 3F12.2)') '  - End   (x1, x2, x3):', x1_end, x2_end, x3_end
            write(*, '(A, 3F12.2)') '  - Size  (L1, L2, L3):', L1, L2, L3
            write(*, '(A, F12.2)')  '  - Volume:', Volume
            write(*,*)
        
            write(*, '(A)') '2 - Mesh Settings:'
            write(*, '(A, 3I12)') '  - Grid Points (n1m, n2m, n3m):', n1m, n2m, n3m
            write(*, '(A, 3I12)') '  - Plus Points (n1p, n2p, n3p):', n1p, n2p, n3p
            write(*,*)
        
            write(*, '(A)') '3 - Grid Style:'
            write(*, '(A, 3L12)')   '  - PBC (pbc1, pbc2, pbc3)                   : ', pbc1, pbc2, pbc3
            write(*, '(A, 3A12)')   '  - Uniformity (uniform1, uniform2, uniform3): ', uniform_type(uniform1), uniform_type(uniform2), uniform_type(uniform3)
            write(*, '(A, 3F12.2)') '  - Gamma (gamma1, gamma2, gamma3)           : ', gamma1, gamma2, gamma3
            write(*,*)
        
            write(*, '(A)') '4 - Parallel Settings:'
            write(*, '(A, 3I12)') '  - MPI Procs (np1, np2, np3):', np1, np2, np3
            write(*, '(A, 3I12)') '  - CUDA Thread (x, y, z)    :', thread_in_x, thread_in_y, thread_in_z
            write(*, '(A, 3I12)') '  - CUDA Pascal TDMA Thread  :', thread_in_x_pascal, thread_in_y_pascal, thread_in_fft_pascal
            write(*,*)
        
            write(*, '(A, A20)') '  - Poisson Equation (Poisson_model):', trim(poisson_type(Poisson_model))
        
            write(*, '(A)') '8 - Post Options:'
            write(*, '(A, 2L12)') '  - Continue File in/out (ContinueFilein, ContinueFileout)     :', ContinueFilein, ContinueFileout
            write(*,*)

            write(*, '(A)') '9 - Directories:'
            write(*, '(A, A)') '  - Infile        : ', trim(dir_cont_filein)
            write(*, '(A, A)') '  - Outfile       : ', trim(dir_cont_fileout)
            write(*,*)

            write(*, '(A)') '================================================================================'
        endif        

        ! Grid check for Poisson FFT communication requirements and even number constraint
        if (n1m < (2 * np1 * np3 - 2)) then
            write(*,*) "n1m should be greater than or equal to 2*np1*np3 - 2 for Poisson FFT All-to-All communication. Program will be terminated."
            call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
        elseif (mod(n1m, 2) /= 0 .or. mod(n2m, 2) /= 0 .or. mod(n3m, 2) /= 0) then
            write(*,*) "n1m, n2m, and n3m must all be even numbers. Program will be terminated."
            call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
        endif
        write(*,*)

        ! Folder generation
        call system('mkdir -p ./data'); call system('mkdir -p ./data/1_continue'); call system('mkdir -p ./data/2_instantfield'); 

    end subroutine global_print_settings


    function uniform_type(uniform) result(uniformString)
        implicit none
        integer, intent(in) :: uniform
        character(len=20) :: uniformString
        
        select case (uniform)
        case (0)
            uniformString = 'Non-uniform'
        case (1)
            uniformString = 'Uniform'
        case (2)
            uniformString = 'Non-uniform-2'
        case (3)
            uniformString = 'Stretch_grid'
        case default
            uniformString = 'Unknown'
        end select
    end function uniform_type

    function poisson_type(poisson) result(poissonString)
        implicit none
        integer, intent(in) :: poisson
        character(len=20) :: poissonString
        
        select case (poisson)
        case (1)
            poissonString = 'FFT'
        case (2)
            poissonString = 'Multigrid'
        case default
            poissonString = 'Unknown'
        end select
    end function poisson_type

end module global
