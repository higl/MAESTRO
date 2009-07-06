! diagnostic routine for the flame propagation problem

module diag_module

  use bl_types
  use bl_IO_module
  use multifab_module
  use ml_layout_module
  use define_bc_module

  implicit none

  private

  public :: diag

contains

  subroutine diag(time,dt,dx,s,rho_Hnuc,rho_Hext,rho_omegadot, &
                  rho0,rhoh0,p0,tempbar, &
                  gamma1bar,div_coeff, &
                  u,w0,normal, &
                  mla,the_bc_tower)

    use bl_prof_module
    use geometry, only: dm, nlevs, spherical
    use bl_constants_module
    use probin_module, only: prob_lo_x, prob_lo_y, prob_lo_z, &
                             prob_hi_x, prob_hi_y, prob_hi_z, &
                             job_name
    use inlet_bc_module

    real(kind=dp_t), intent(in   ) :: dt,dx(:,:),time
    type(multifab) , intent(in   ) :: s(:)
    type(multifab) , intent(in   ) :: rho_Hnuc(:)
    type(multifab) , intent(in   ) :: rho_Hext(:)
    type(multifab) , intent(in   ) :: rho_omegadot(:)
    type(multifab) , intent(in   ) :: u(:)
    type(multifab) , intent(in   ) :: normal(:)
    real(kind=dp_t), intent(in   ) ::      rho0(:,0:)
    real(kind=dp_t), intent(in   ) ::     rhoh0(:,0:)
    real(kind=dp_t), intent(in   ) ::        p0(:,0:)
    real(kind=dp_t), intent(in   ) ::   tempbar(:,0:)
    real(kind=dp_t), intent(in   ) :: gamma1bar(:,0:)
    real(kind=dp_t), intent(in   ) :: div_coeff(:,0:)
    real(kind=dp_t), intent(in   ) ::        w0(:,0:)
    type(ml_layout), intent(in   ) :: mla
    type(bc_tower) , intent(in   ) :: the_bc_tower

    ! Local
    real(kind=dp_t), pointer::  sp(:,:,:,:)
    real(kind=dp_t), pointer::  rhnp(:,:,:,:)
    real(kind=dp_t), pointer::  rhep(:,:,:,:)
    real(kind=dp_t), pointer::  rwp(:,:,:,:)
    real(kind=dp_t), pointer::  up(:,:,:,:)
    real(kind=dp_t), pointer::  np(:,:,:,:)
    logical        , pointer::  mp(:,:,:,:)

    real(kind=dp_t) :: flame_speed, flame_speed_level, flame_speed_local
    
    integer :: lo(dm),hi(dm)
    integer :: ng_s,ng_u,ng_n,ng_rhn,ng_rhe,ng_rw
    integer :: i,n
    integer :: un
    logical :: lexist

    integer, save :: ic12
    logical, save :: firstCall_io = .true.
    logical, save :: firstCall_params = .true.

    type(bl_prof_timer), save :: bpt

    call build(bpt, "diagnostics")

    if (firstCall_params) then

       ic12 = network_species_index("carbon-12")

       firstCall_params = .false.
    endif

    ng_s = s(1)%ng
    ng_u = u(1)%ng
    ng_n = normal(1)%ng
    ng_rhn = rho_Hnuc(1)%ng
    ng_rhe = rho_Hext(1)%ng
    ng_rw = rho_omegadot(1)%ng

    !=========================================================================
    ! initialize
    !=========================================================================
    flame_speed = ZERO

    !=========================================================================
    ! loop over the levels and compute the global quantities
    !=========================================================================
    do n = 1, nlevs

       ! initialize the local (processor's version) and level quantities to 0
       flame_speed_level = ZERO
       flame_speed_local = ZERO

       !----------------------------------------------------------------------
       ! loop over boxes in a given level
       !----------------------------------------------------------------------
       do i = 1, s(n)%nboxes
          if ( multifab_remote(s(n), i) ) cycle
          sp => dataptr(s(n) , i)
          rhnp => dataptr(rho_Hnuc(n), i)
          rhep => dataptr(rho_Hext(n), i)
          rwp => dataptr(rho_omegadot(n), i)
          up => dataptr(u(n) , i)
          lo =  lwb(get_box(s(n), i))
          hi =  upb(get_box(s(n), i))
          select case (dm)
          case (2)
             if (n .eq. nlevs) then
                call diag_2d(n,time,dt,dx(n,:), &
                             sp(:,:,1,:),ng_s, &
                             rhnp(:,:,1,1),ng_rhn, &
                             rhep(:,:,1,1),ng_rhe, &
                             rwp(:,:,1,:),ng_rw, &
                             rho0(n,:),rhoh0(n,:), &
                             p0(n,:),tempbar(n,:),gamma1bar(n,:), &
                             up(:,:,1,:),ng_u, &
                             w0(n,:), &
                             lo,hi, &
                             flame_speed_local)
             else
                mp => dataptr(mla%mask(n), i)
                call diag_2d(n,time,dt,dx(n,:), &
                             sp(:,:,1,:),ng_s, &
                             rhnp(:,:,1,1),ng_rhn, &
                             rhep(:,:,1,1),ng_rhe, &
                             rwp(:,:,1,:),ng_rw, &
                             rho0(n,:),rhoh0(n,:), &
                             p0(n,:),tempbar(n,:),gamma1bar(n,:), &
                             up(:,:,1,:),ng_u, &
                             w0(n,:), &
                             lo,hi, &
                             flame_speed_local, &
                             mp(:,:,1,1))
             endif
          case (3)
             call bl_error("ERROR: 3D diagnostics not implemented")
          end select
       end do

       !----------------------------------------------------------------------
       ! do the appropriate parallel reduction for the current level
       !----------------------------------------------------------------------

       ! NOTE: only the I/O Processor will have the correct reduced value
       call parallel_reduce(flame_speed_level, flame_speed_local, MPI_SUM, &
                            proc = parallel_IOProcessorNode())

       !----------------------------------------------------------------------
       ! reduce the current level's data with the global data
       !----------------------------------------------------------------------
       if (parallel_IOProcessor()) then

          flame_speed = flame_speed + flame_speed_level

       endif

    end do

    !=========================================================================
    ! normalize
    !=========================================================================
    
    ! our flame speed estimate is based on the carbon destruction rate
    ! V_eff = - int { rho omegadot dx } / W (rho X)^in
    flame_speed = -flame_speed*dx(1,1)*dx(1,2)/ &
         ((prob_hi_x - prob_lo_x)*inlet_rhox(ic12))
    
    

    !=========================================================================
    ! output
    !=========================================================================
 999 format("# job name: ",a)
1000 format(1x,10(g20.10,1x))
1001 format("#",10(a20,1x))

    if (parallel_IOProcessor()) then

       ! open the diagnostic files for output, taking care not to overwrite
       ! an existing file
       un = unit_new()
       inquire(file="flame_basic_diag.out", exist=lexist)
       if (lexist) then
          open(unit=un, file="flame_basic_diag.out", &
               status="old", position="append")
       else
          open(unit=un, file="flame_basic_diag.out", status="new")
       endif


       ! write out the headers
       if (firstCall_io) then

          ! radvel
          write (un, *) " "
          write (un, 999) trim(job_name)
          write (un, 1001) "time", "flame speed", "flame thickness"

          firstCall_io = .false.
       endif

       ! write out the data
       write (un,1000) time, flame_speed

       close(un)

    endif

    call destroy(bpt)

  end subroutine diag

  subroutine diag_2d(n,time,dt,dx, &
                     s,ng_s, &
                     rho_Hnuc,ng_rhn, &
                     rho_Hext,ng_rhe, &
                     rho_omegadot,ng_rw, &
                     rho0,rhoh0,p0,tempbar,gamma1bar, &
                     u,ng_u, &
                     w0, &
                     lo,hi, &
                     flame_speed, &
                     mask)

    use variables, only: rho_comp, spec_comp, temp_comp, rhoh_comp
    use bl_constants_module
    use network, only: nspec, network_species_index
    use probin_module, only: prob_lo

    integer, intent(in) :: n, lo(:), hi(:), ng_s, ng_u, ng_rhn, ng_rhe, ng_rw
    real (kind=dp_t), intent(in   ) ::      s(lo(1)-ng_s:,lo(2)-ng_s:,:)
    real (kind=dp_t), intent(in   ) :: rho_Hnuc(lo(1)-ng_rhn:,lo(2)-ng_rhn:)
    real (kind=dp_t), intent(in   ) :: rho_Hext(lo(1)-ng_rhe:,lo(2)-ng_rhe:)
    real (kind=dp_t), intent(in   ) :: rho_omegadot(lo(1)-ng_rw:,lo(2)-ng_rw:,:)
    real (kind=dp_t), intent(in   ) :: rho0(0:), rhoh0(0:), &
                                         p0(0:),tempbar(0:),gamma1bar(0:)
    real (kind=dp_t), intent(in   ) ::      u(lo(1)-ng_u:,lo(2)-ng_u:,:)
    real (kind=dp_t), intent(in   ) :: w0(0:)
    real (kind=dp_t), intent(in   ) :: time, dt, dx(:)
    real (kind=dp_t), intent(inout) :: flame_speed
    logical,          intent(in   ), optional :: mask(lo(1):,lo(2):)

    !     Local variables
    integer            :: i, j
    real (kind=dp_t)   :: weight
    logical            :: cell_valid
    real (kind=dp_t)   :: x, y

    integer, save :: ic12

    logical, save :: firstCall = .true.


    if (firstCall) then

       ic12 = network_species_index("carbon-12")

       firstCall = .false.
    endif

    ! weight is the factor by which the volume of a cell at the current level
    ! relates to the volume of a cell at the coarsest level of refinement.
    weight = 1.d0 / 4.d0**(n-1)

    do j = lo(2), hi(2)
       y = prob_lo(2) + (dble(j) + HALF) * dx(2)
       
       do i = lo(1), hi(1)
          x = prob_lo(1) + (dble(i) + HALF) * dx(1)

          cell_valid = .true.
          if (present(mask)) then
             if ( (.not. mask(i,j)) ) cell_valid = .false.
          endif

          if (cell_valid) then

             ! compute the flame speed by integrating the carbon
             ! consumption rate.
             flame_speed = flame_speed + weight*rho_omegadot(i,j,ic12)
             
             ! compute the min/max temperature and the maximum
             ! temperature gradient for computing the flame thickness.


          endif

       enddo
    enddo

  end subroutine diag_2d

end module diag_module
