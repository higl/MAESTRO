! given a multifab of data (phi), average down to a base state quantity,
! phibar.  If we are in plane-parallel, the averaging is at constant
! height.  If we are spherical, then the averaging is done at constant
! radius.  

module average_module
  
  use bl_types
  use multifab_module
  use ml_layout_module

  implicit none

  private
  public :: average

contains

  subroutine average(mla,phi,phibar,dx,incomp)

    use geometry, only: nr_fine, r_start_coord, r_end_coord, spherical, center, dr
    use bl_prof_module
    use bl_constants_module

    type(ml_layout), intent(in   ) :: mla
    integer        , intent(in   ) :: incomp
    type(multifab) , intent(inout) :: phi(:)   ! Need the out so layout_aveassoc() can 
                                               ! modify the layout.
    real(kind=dp_t), intent(inout) :: phibar(:,0:)
    real(kind=dp_t), intent(in   ) :: dx(:,:)

    ! local
    real(kind=dp_t), pointer     :: pp(:,:,:,:)
    logical, pointer             :: mp(:,:,:,:)
    type(box)                    :: domain
    integer                      :: domlo(phi(1)%dim),domhi(phi(1)%dim)
    integer                      :: lo(phi(1)%dim),hi(phi(1)%dim)
    integer                      :: i,r,n,nlevs,ng,dm,rr,nsub
    real(kind=dp_t), allocatable :: ncell_grid(:,:)
    real(kind=dp_t), allocatable :: ncell_proc(:,:)
    real(kind=dp_t), allocatable :: ncell(:,:)
    real(kind=dp_t), allocatable :: phisum_proc(:,:)
    real(kind=dp_t), allocatable :: phisum(:,:)
    real(kind=dp_t), allocatable :: phipert_proc(:,:)
    real(kind=dp_t), allocatable :: phipert(:,:)
    real(kind=dp_t), allocatable :: source_buffer(:)
    real(kind=dp_t), allocatable :: target_buffer(:)
    logical                      :: fine_grids_span_domain_width

    type(aveassoc) :: avasc

    type(bl_prof_timer), save :: bpt

    call build(bpt, "average")

    dm = phi(1)%dim
    ng = phi(1)%ng
    nlevs = size(dx,dim=1)

    phibar = ZERO

    if (spherical .eq. 1) then
       allocate(ncell_grid(nlevs,0:nr_fine-1))
    end if

    allocate(ncell_proc   (nlevs,0:nr_fine-1))
    allocate(ncell        (nlevs,0:nr_fine-1))
    allocate(phisum_proc  (nlevs,0:nr_fine-1))
    allocate(phisum       (nlevs,0:nr_fine-1))
    allocate(phipert_proc (nlevs,0:nr_fine-1))
    allocate(phipert      (nlevs,0:nr_fine-1))
    allocate(source_buffer      (0:nr_fine-1))
    allocate(target_buffer      (0:nr_fine-1))

    ncell        = ZERO
    ncell_proc   = ZERO
    phisum       = ZERO       
    phisum_proc  = ZERO
    phipert      = ZERO
    phipert_proc = ZERO

    if (spherical .eq. 0) then

       fine_grids_span_domain_width = .true.

       if (fine_grids_span_domain_width) then

          do n=1,nlevs

             domain = layout_get_pd(phi(n)%la)
             domlo  = lwb(domain)
             domhi  = upb(domain)
             
             if (dm .eq. 2) then
                ncell(n,:) = domhi(1)-domlo(1)+1
             else if (dm .eq. 3) then
                ncell(n,:) = (domhi(1)-domlo(1)+1)*(domhi(2)-domlo(2)+1)
             end if
             
             ! the first step is to compute phibar assuming the coarsest level 
             ! is the only level in existence
             do i=1,phi(n)%nboxes
                if ( multifab_remote(phi(n), i) ) cycle
                pp => dataptr(phi(n), i)
                lo =  lwb(get_box(phi(n), i))
                hi =  upb(get_box(phi(n), i))
                select case (dm)
                case (2)
                   call sum_phi_coarsest_2d(pp(:,:,1,:),phisum_proc(n,:),lo,hi,ng,incomp)
                case (3)
                   call sum_phi_coarsest_3d(pp(:,:,:,:),phisum_proc(n,:),lo,hi,ng,incomp)
                end select
             end do
             
             ! gather phisum
             source_buffer = phisum_proc(n,:)
             call parallel_reduce(target_buffer, source_buffer, MPI_SUM)
             phisum(n,:) = target_buffer
             do r=r_start_coord(n),r_end_coord(n)
                phibar(n,r) = phisum(n,r) / dble(ncell(n,r))
             end do
             
          end do

       else

          domain = layout_get_pd(phi(1)%la)
          domlo  = lwb(domain)
          domhi  = upb(domain)
          
          if (dm .eq. 2) then
             ncell(1,:) = domhi(1)-domlo(1)+1
          else if (dm .eq. 3) then
             ncell(1,:) = (domhi(1)-domlo(1)+1)*(domhi(2)-domlo(2)+1)
          end if
          
          ! the first step is to compute phibar assuming the coarsest level 
          ! is the only level in existence
          do i=1,phi(1)%nboxes
             if ( multifab_remote(phi(1), i) ) cycle
             pp => dataptr(phi(1), i)
             lo =  lwb(get_box(phi(1), i))
             hi =  upb(get_box(phi(1), i))
             select case (dm)
             case (2)
                call sum_phi_coarsest_2d(pp(:,:,1,:),phisum_proc(1,:),lo,hi,ng,incomp)
             case (3)
                call sum_phi_coarsest_3d(pp(:,:,:,:),phisum_proc(1,:),lo,hi,ng,incomp)
             end select
          end do
          
          ! gather phisum
          source_buffer = phisum_proc(1,:)
          call parallel_reduce(target_buffer, source_buffer, MPI_SUM)
          phisum(1,:) = target_buffer
          do r=0,r_end_coord(1)
             phibar(1,r) = phisum(1,r) / dble(ncell(1,r))
          end do
          
          ! now we compute the phibar at the finer levels
          do n=2,nlevs
             
             rr = mla%mba%rr(n-1,dm)
             
             domain = layout_get_pd(phi(n)%la)
             domlo  = lwb(domain)
             domhi  = upb(domain)
             
             if (dm .eq. 2) then
                ncell(n,:) = domhi(1)-domlo(1)+1
             else if (dm .eq. 3) then
                ncell(n,:) = (domhi(1)-domlo(1)+1)*(domhi(2)-domlo(2)+1)
             end if
             
             ! compute phisum at next finer level
             ! begin by assuming piecewise constant interpolation
             do r=r_start_coord(n),r_end_coord(n)
                phisum(n,r) = phisum(n-1,r/rr)*rr**(dm-1)
             end do
             
             ! compute phipert_proc
             do i=1,phi(n)%nboxes
                if ( multifab_remote(phi(n), i) ) cycle
                pp => dataptr(phi(n), i)
                lo =  lwb(get_box(phi(n), i))
                hi =  upb(get_box(phi(n), i))
                select case (dm)
                case (2)
                   call compute_phipert_2d(pp(:,:,1,:),phipert_proc(n,:),lo,hi,ng,incomp,rr)
                case (3)
                   call compute_phipert_3d(pp(:,:,:,:),phipert_proc(n,:),lo,hi,ng,incomp,rr)
                end select
             end do
             
             ! gather phipert
             source_buffer = phipert_proc(n,:)
             call parallel_reduce(target_buffer, source_buffer, MPI_SUM)
             phipert(n,:) = target_buffer
             
             ! update phisum and compute phibar
             do r=r_start_coord(n),r_end_coord(n)
                phisum(n,r) = phisum(n,r) + phipert(n,r)
                phibar(n,r) = phisum(n,r) / dble(ncell(n,r))
             end do

          end do

       end if

    else if(spherical .eq. 1) then

       ! The spherical case is tricky because the base state only exists at one level
       ! as defined by dr_base in the inputs file.
       ! Therefore, the goal here is to compute phibar(nlevs,:).
       ! phisum(nlevs,:,:) will be the volume weighted sum over all levels.
       ! ncell(nlevs,:) will be the volume weighted number of cells over all levels.
       ! We make sure to use mla%mask to not double count cells, i.e.,
       ! we only sum up cells that are not covered by finer cells.
       ! we use the convention that a cell volume of 1 corresponds to dx(n=1)**3

       ! First we compute ncell(nlevs,:) and phisum(nlevs,:,:) as if the finest level
       ! were the only level in existence.
       ! Then, we add contributions from each coarser cell that is not covered by 
       ! a finer cell.

       do n=nlevs,1,-1

          ! This MUST match the nsub in average_3d_sphr().
          nsub = int(dx(n,1)/dr(nlevs)) + 1  

          avasc = layout_aveassoc(phi(n)%la,nsub,phi(n)%nodal,dx(n,:),center,dr(nlevs))

          do i=1,phi(n)%nboxes
             if ( multifab_remote(phi(n), i) ) cycle
             pp => dataptr(phi(n), i)
             lo =  lwb(get_box(phi(n), i))
             hi =  upb(get_box(phi(n), i))
             ncell_grid(n,:) = ZERO
             if (n .eq. nlevs) then
                call average_3d_sphr(n,nlevs,pp(:,:,:,:),phisum_proc(n,:),avasc%fbs(i), &
                                     lo,hi,ng,dx(n,:),ncell_grid(n,:),incomp,mla)
             else
                mp => dataptr(mla%mask(n), i)
                call average_3d_sphr(n,nlevs,pp(:,:,:,:),phisum_proc(n,:),avasc%fbs(i), &
                                     lo,hi,ng,dx(n,:),ncell_grid(n,:),incomp,mla, &
                                     mp(:,:,:,1))
!                call average_3d_sphr_linear(n,nlevs,pp(:,:,:,:),phisum_proc(n,:), &
!                                            avasc%fbs(i),lo,hi,ng,dx(n,:),ncell_grid(n,:), &
!                                            incomp,mla,mp(:,:,:,1))           
             end if

             ncell_proc(n,:) = ncell_proc(n,:) + ncell_grid(n,:)
          end do

          call parallel_reduce(ncell(n,:), ncell_proc(n,:), MPI_SUM)

          source_buffer = phisum_proc(n,:)
          call parallel_reduce(target_buffer, source_buffer, MPI_SUM)
          phisum(n,:) = target_buffer

          if (n .ne. nlevs) then
             ncell(nlevs,:) = ncell(nlevs,:) + ncell(n,:)
             do r=r_start_coord(n),r_end_coord(n)
                phisum(nlevs,r) = phisum(nlevs,r) + phisum(n,r)
             end do
          end if

       end do

       ! now divide the total phisum by the number of cells to get phibar
       do r=r_start_coord(nlevs),r_end_coord(nlevs)
          if (ncell(nlevs,r) .gt. ZERO) then
             phibar(nlevs,r) = phisum(nlevs,r) / ncell(nlevs,r)
          else
             phibar(nlevs,r) = ZERO
          end if
       end do
       
       ! temporary hack for the case where the outermost radial bin average to zero
       ! because there is no contribution from any Cartesian cell that lies in this bin.
       ! this needs to be addressed - perhaps in the definition of r_end_coord in varden.f90
       ! for spherical problems.
       if (ncell(nlevs,r_end_coord(n)) .eq. ZERO) then
          phibar(nlevs,r_end_coord(n)) = phibar(nlevs,r_end_coord(n)-1)
       end if

       deallocate(ncell_grid)

       deallocate(ncell_proc,ncell)
       deallocate(phisum_proc,phisum)
       deallocate(phipert_proc,phipert)
       deallocate(source_buffer,target_buffer)

    endif

    call destroy(bpt)

  end subroutine average

  subroutine sum_phi_coarsest_2d(phi,phisum,lo,hi,ng,incomp)

    integer         , intent(in   ) :: lo(:), hi(:), ng, incomp
    real (kind=dp_t), intent(in   ) :: phi(lo(1)-ng:,lo(2)-ng:,:)
    real (kind=dp_t), intent(inout) :: phisum(0:)

    integer :: i,j

    do j=lo(2),hi(2)
       do i=lo(1),hi(1)
          phisum(j) = phisum(j) + phi(i,j,incomp)
       end do
    end do

  end subroutine sum_phi_coarsest_2d

  subroutine sum_phi_coarsest_3d(phi,phisum,lo,hi,ng,incomp)

    integer         , intent(in   ) :: lo(:), hi(:), ng, incomp
    real (kind=dp_t), intent(in   ) :: phi(lo(1)-ng:,lo(2)-ng:,lo(3)-ng:,:)
    real (kind=dp_t), intent(inout) :: phisum(0:)

    integer :: i,j,k

    do k=lo(3),hi(3)
       do j=lo(2),hi(2)
          do i=lo(1),hi(1)
             phisum(k) = phisum(k) + phi(i,j,k,incomp)
          end do
       end do
    end do

  end subroutine sum_phi_coarsest_3d

  subroutine compute_phipert_2d(phi,phipert,lo,hi,ng,incomp,rr)

    use bl_constants_module

    integer         , intent(in   ) :: lo(:), hi(:), ng, incomp, rr
    real (kind=dp_t), intent(in   ) :: phi(lo(1)-ng:,lo(2)-ng:,:)
    real (kind=dp_t), intent(inout) :: phipert(0:)

    ! Local variables
    integer          :: i, j, icrse, jcrse
    real (kind=dp_t) :: crseval

    ! loop over coarse cell index
    do jcrse=lo(2)/rr,hi(2)/rr
       do icrse=lo(1)/rr,hi(1)/rr
          
          crseval = ZERO
          
          ! compute coarse cell value by taking average of fine cells
          do j=0,rr-1
             do i=0,rr-1
                crseval = crseval + phi(icrse*rr+i,jcrse*rr+j,incomp)
             end do
          end do
          crseval = crseval / dble(rr**2)
          
          ! compute phipert
          do j=0,rr-1
             do i=0,rr-1
                phipert(jcrse*rr+j) = phipert(jcrse*rr+j) &
                     + phi(icrse*rr+i,jcrse*rr+j,incomp) - crseval
             end do
          end do
          
       end do
    end do

  end subroutine compute_phipert_2d

  subroutine compute_phipert_3d(phi,phipert,lo,hi,ng,incomp,rr)

    use bl_constants_module

    integer         , intent(in   ) :: lo(:), hi(:), ng, incomp, rr
    real (kind=dp_t), intent(in   ) :: phi(lo(1)-ng:,lo(2)-ng:,lo(3)-ng:,:)
    real (kind=dp_t), intent(inout) :: phipert(0:)

    ! Local variables
    integer          :: i, j, k, icrse, jcrse, kcrse
    real (kind=dp_t) :: crseval

    ! loop over coarse cell index
    do kcrse=lo(3)/rr,hi(3)/rr
       do jcrse=lo(2)/rr,hi(2)/rr
          do icrse=lo(1)/rr,hi(1)/rr
             
             crseval = ZERO
             
             ! compute coarse cell value by taking average of fine cells
             do k=0,rr-1
                do j=0,rr-1
                   do i=0,rr-1
                      crseval = crseval + phi(icrse*rr+i,jcrse*rr+j,kcrse*rr+k,incomp)
                   end do
                end do
             end do
             crseval = crseval / dble(rr**3)
             
             ! compute phipert
             do k=0,rr-1
                do j=0,rr-1
                   do i=0,rr-1
                      phipert(kcrse*rr+k) = phipert(kcrse*rr+k) &
                           + phi(icrse*rr+i,jcrse*rr+j,kcrse*rr+k,incomp) - crseval
                   end do
                end do
             end do
             
          end do
       end do
    end do
    
  end subroutine compute_phipert_3d

  subroutine average_3d_sphr(n,nlevs,phi,phisum,avfab,lo,hi,ng,dx,ncell,incomp,mla,mask)

    use geometry, only: spherical, dr, center
    use ml_layout_module
    use bl_constants_module

    integer         , intent(in   ) :: n, nlevs
    integer         , intent(in   ) :: lo(:), hi(:), ng, incomp
    real (kind=dp_t), intent(in   ) :: phi(lo(1)-ng:,lo(2)-ng:,lo(3)-ng:,:)
    type(avefab)    , intent(in   ) :: avfab
    real (kind=dp_t), intent(inout) :: phisum(0:)
    real (kind=dp_t), intent(in   ) :: dx(:)
    real (kind=dp_t), intent(inout) :: ncell(0:)
    type(ml_layout) , intent(in   ) :: mla
    logical         , intent(in   ), optional :: mask(lo(1):,lo(2):,lo(3):)

    integer          :: i, j, k, l, idx, cnt, nsub
    real (kind=dp_t) :: cell_weight
    logical          :: cell_valid
    !
    ! Compute nsub such that we are always guaranteed to fill each of
    ! the base state radial bins.
    !
    nsub = int(dx(1)/dr(nlevs)) + 1

    cell_weight = 1.d0 / nsub**3
    do i=2,n
       cell_weight = cell_weight / (mla%mba%rr(i-1,1))**3
    end do

    do k=lo(3),hi(3)
       do j=lo(2),hi(2)
          do i=lo(1),hi(1)
             cell_valid = .true.
             if ( present(mask) ) then
                if ( (.not. mask(i,j,k)) ) cell_valid = .false.
             end if
             if (cell_valid) then
                do l=1,size(avfab%p(i,j,k)%v,dim=1)
                   idx = avfab%p(i,j,k)%v(l,1)
                   cnt = avfab%p(i,j,k)%v(l,2)
                   phisum(idx) = phisum(idx) + cnt*cell_weight*phi(i,j,k,incomp)
                   ncell(idx) = ncell(idx) + cnt*cell_weight
                end do
             end if
          end do
       end do
    end do

  end subroutine average_3d_sphr

  subroutine average_3d_sphr_linear(n,nlevs,phi,phisum,avfab,lo,hi,ng,dx,ncell,incomp, &
                                    mla,mask)

    use geometry, only: spherical, dr, center
    use ml_layout_module
    use bl_constants_module

    integer         , intent(in   ) :: n, nlevs
    integer         , intent(in   ) :: lo(:), hi(:), ng, incomp
    real (kind=dp_t), intent(in   ) :: phi(lo(1)-ng:,lo(2)-ng:,lo(3)-ng:,:)
    type(avefab)    , intent(in   ) :: avfab
    real (kind=dp_t), intent(inout) :: phisum(0:)
    real (kind=dp_t), intent(in   ) :: dx(:)
    real (kind=dp_t), intent(inout) :: ncell(0:)
    type(ml_layout) , intent(in   ) :: mla
    logical         , intent(in   ), optional :: mask(lo(1):,lo(2):,lo(3):)

    integer          :: i, j, k, l, idx, nsub
    real (kind=dp_t) :: cell_weight
    logical          :: cell_valid
    real (kind=dp_t) :: xl, yl, zl, xc, yc, zc, x, y, z, radius
    real (kind=dp_t) :: m_x, m_y, m_z, test, val

    integer :: ii, jj, kk

    !
    ! Compute nsub such that we are always guaranteed to fill each of
    ! the base state radial bins.
    !
    nsub = int(dx(1)/dr(nlevs)) + 1

    cell_weight = 1.d0 / nsub**3
    do i=2,n
       cell_weight = cell_weight / (mla%mba%rr(i-1,1))**3
    end do

    do k=lo(3),hi(3)
       zl = dble(k)*dx(3) - center(3)
       zc = (dble(k) + HALF)*dx(3) - center(3)

       do j=lo(2),hi(2)
          yl = dble(j)*dx(2) - center(2)
          yc = (dble(j) + HALF)*dx(2) - center(2)

          do i=lo(1),hi(1)
             xl = dble(i)*dx(1) - center(1)
             xc = (dble(i) + HALF)*dx(1) - center(1)

             cell_valid = .true.
             if ( present(mask) ) then
                if ( (.not. mask(i,j,k)) ) cell_valid = .false.
             end if

             ! use linear reconstruction on the data -- start by constructing
             ! the slopes (limiter ref. Collela 1990, Eq. 1.9)

             ! x-slope
             test = (phi(i+1,j,k,incomp) - phi(i,  j,k,incomp))* &
                    (phi(i,  j,k,incomp) - phi(i-1,j,k,incomp))

             if (test > ZERO) then
                m_x = min(HALF*abs(phi(i+1,j,k,incomp) - phi(i-1,j,k,incomp)), &
                          min(TWO*abs(phi(i+1,j,k,incomp) - phi(i,  j,k,incomp)), &
                              TWO*abs(phi(i,  j,k,incomp) - phi(i-1,j,k,incomp)))) * &
                              sign(ONE,phi(i+1,j,k,incomp) - phi(i-1,j,k,incomp))
             else
                m_x = ZERO
             endif

             ! y-slope
             test = (phi(i,j+1,k,incomp) - phi(i,j,  k,incomp))* &
                    (phi(i,j,  k,incomp) - phi(i,j-1,k,incomp))

             if (test > ZERO) then
                m_y = min(HALF*abs(phi(i,j+1,k,incomp) - phi(i,j-1,k,incomp)), &
                          min(TWO*abs(phi(i,j+1,k,incomp) - phi(i,j,  k,incomp)), &
                              TWO*abs(phi(i,j,  k,incomp) - phi(i,j-1,k,incomp)))) * &
                              sign(ONE,phi(i,j+1,k,incomp) - phi(i,j-1,k,incomp))
             else
                m_y = ZERO
             endif

             ! z-slope
             test = (phi(i,j,k+1,incomp) - phi(i,j,k,  incomp))* &
                    (phi(i,j,k  ,incomp) - phi(i,j,k-1,incomp))

             if (test > ZERO) then
                m_z = min(HALF*abs(phi(i,j,k+1,incomp) - phi(i,j,k-1,incomp)), &
                          min(TWO*abs(phi(i,j,k+1,incomp) - phi(i,j,k  ,incomp)), &
                              TWO*abs(phi(i,j,k  ,incomp) - phi(i,j,k-1,incomp)))) * &
                              sign(ONE,phi(i,j,k+1,incomp) - phi(i,j,k-1,incomp))
             else
                m_z = ZERO
             endif
             

             if (cell_valid) then

                do kk = 0, nsub-1
                   z = dble(kk+HALF)*dx(3)/nsub + zl

                   do jj = 0, nsub-1
                      y = dble(jj+HALF)*dx(2)/nsub + yl

                      do ii = 0, nsub-1
                         x = dble(ii+HALF)*dx(1)/nsub + xl

                         radius = sqrt(x**2 + y**2 + z**2)
                         idx  = int(radius / dr(n))

                         val = m_x*(x - xc)/dx(1) + m_y*(y - yc)/dx(2) + m_z*(z - zc)/dx(3) + phi(i,j,k,incomp)

                         phisum(idx) = phisum(idx) + cell_weight*val
                         ncell(idx) = ncell(idx) + cell_weight

                      enddo
                   enddo
                enddo

             end if

          end do
       end do
    end do

  end subroutine average_3d_sphr_linear

end module average_module
