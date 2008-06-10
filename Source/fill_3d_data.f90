module fill_3d_module

  use bl_types
  use multifab_module

  implicit none

  private
  
  public :: put_1d_array_on_cart, put_1d_array_on_cart_3d_sphr, put_1d_array_on_edges
  public :: make_3d_normal
  
contains  

  subroutine put_1d_array_on_cart(nlevs,s0,s0_cart,bc_comp,is_edge_centered,is_vector, &
                                  dx,the_bc_level,mla,normal)

    use bl_prof_module
    use bl_constants_module
    use define_bc_module
    use geometry, only: spherical
    use ml_layout_module
    use multifab_physbc_module
    use ml_restriction_module, only: ml_cc_restriction_c
    use multifab_fill_ghost_module
    use variables, only: foextrap_comp
    
    integer        , intent(in   ) :: nlevs
    real(kind=dp_t), intent(in   ) :: s0(:,0:)
    type(multifab) , intent(inout) :: s0_cart(:)
    integer        , intent(in   ) :: bc_comp
    logical        , intent(in   ) :: is_edge_centered,is_vector
    real(kind=dp_t), intent(in   ) :: dx(:,:)
    type(bc_level) , intent(in   ) :: the_bc_level(:)
    type(ml_layout), intent(in   ) :: mla
    type(multifab) , intent(in   ), optional :: normal(:)
    
    integer :: lo(s0_cart(1)%dim)
    integer :: hi(s0_cart(1)%dim)
    integer :: i,n,dm,ng,comp
    real(kind=dp_t), pointer :: sp(:,:,:,:)
    real(kind=dp_t), pointer :: np(:,:,:,:)

    type(bl_prof_timer), save :: bpt

    call build(bpt, "put_1d_array_on_cart")

    if (spherical .eq. 1 .and. is_vector .and. (.not. present(normal)) ) then
       call bl_error('Error: Calling put_1d_array_on_cart for spherical with is_vector=T and without normal')
    end if

    dm = s0_cart(1)%dim
    ng = s0_cart(1)%ng
    
    do n=1,nlevs
       
       do i = 1, s0_cart(n)%nboxes
          if ( multifab_remote(s0_cart(n), i) ) cycle
          sp => dataptr(s0_cart(n), i)
          lo =  lwb(get_box(s0_cart(n), i))
          hi =  upb(get_box(s0_cart(n), i))
          select case (dm)

          case (2)
             call put_1d_array_on_cart_2d(n,is_edge_centered,is_vector, &
                                          s0(n,:),sp(:,:,1,:),lo,hi,ng)

          case (3)
             if (spherical .eq. 0) then
                call put_1d_array_on_cart_3d(n,is_edge_centered,is_vector, &
                                             s0(n,:),sp(:,:,:,:),lo,hi,ng)
             else
                if (is_vector) then
                   np => dataptr(normal(n), i)

                   call put_1d_array_on_cart_3d_sphr(n,is_edge_centered,is_vector, &
                                                     s0(n,:),sp(:,:,:,:), &
                                                     lo,hi,dx(n,:),ng,np(:,:,:,:))
                else
                   call put_1d_array_on_cart_3d_sphr(n,is_edge_centered,is_vector, &
                                                     s0(n,:),sp(:,:,:,:), &
                                                     lo,hi,dx(n,:),ng)
                end if
             endif

          end select
       end do

    enddo

    
    if (is_vector) then

       if (bc_comp .eq. foextrap_comp) then

          ! Here we fill each of the dm components using foextrap
          do comp=1,dm
             if (nlevs .eq. 1) then
                call multifab_fill_boundary_c(s0_cart(nlevs),comp,1)
                call multifab_physbc(s0_cart(nlevs),comp,bc_comp,1,the_bc_level(nlevs))
             else
                do n=nlevs,2,-1
                   call ml_cc_restriction_c(s0_cart(n-1),comp,s0_cart(n),comp, &
                                            mla%mba%rr(n-1,:),1)
                   call multifab_fill_ghost_cells(s0_cart(n),s0_cart(n-1),ng, &
                                                  mla%mba%rr(n-1,:),the_bc_level(n-1), &
                                                  the_bc_level(n),comp,bc_comp,1)
                end do
             end if
          end do

       else

          ! Here we fill each of the dm components using bc_comp+comp
          if (nlevs .eq. 1) then
             call multifab_fill_boundary_c(s0_cart(nlevs),1,dm)
             call multifab_physbc(s0_cart(nlevs),1,bc_comp,dm,the_bc_level(nlevs))
          else
             do n=nlevs,2,-1
                call ml_cc_restriction_c(s0_cart(n-1),1,s0_cart(n),1,mla%mba%rr(n-1,:),dm)
                call multifab_fill_ghost_cells(s0_cart(n),s0_cart(n-1),ng, &
                                               mla%mba%rr(n-1,:),the_bc_level(n-1), &
                                               the_bc_level(n),1,bc_comp,dm)
             end do
          end if

       end if

    else

       ! Here will fill the one component using bc_comp
       if (nlevs .eq. 1) then
          call multifab_fill_boundary_c(s0_cart(nlevs),1,1)
          call multifab_physbc(s0_cart(nlevs),1,bc_comp,1,the_bc_level(nlevs))
       else
          do n=nlevs,2,-1
             call ml_cc_restriction_c(s0_cart(n-1),1,s0_cart(n),1,mla%mba%rr(n-1,:),1)
             call multifab_fill_ghost_cells(s0_cart(n),s0_cart(n-1),ng,mla%mba%rr(n-1,:), &
                                            the_bc_level(n-1),the_bc_level(n),1,bc_comp,1)
          end do
       end if

    end if

    call destroy(bpt)
    
  end subroutine put_1d_array_on_cart

  subroutine put_1d_array_on_cart_2d(n,is_edge_centered,is_vector,s0,s0_cart,lo,hi,ng)

    use bl_constants_module
    use geometry, only: dr

    integer        , intent(in   ) :: n
    integer        , intent(in   ) :: lo(:),hi(:),ng
    logical        , intent(in   ) :: is_edge_centered,is_vector
    real(kind=dp_t), intent(in   ) :: s0(0:)
    real(kind=dp_t), intent(inout) :: s0_cart(lo(1)-ng:,lo(2)-ng:,:)

    integer :: i,j

    s0_cart = ZERO

    if (is_edge_centered) then

       if (is_vector) then

          do j=lo(2),hi(2)
             do i=lo(1),hi(1)
                s0_cart(i,j,2) = HALF * (s0(j) + s0(j+1))
             end do
          end do

       else

          do j=lo(2),hi(2)
             do i=lo(1),hi(1)
                s0_cart(i,j,1) = HALF * (s0(j) + s0(j+1))
             end do
          end do

       end if

    else

       if (is_vector) then

          do j=lo(2),hi(2)
             do i=lo(1),hi(1)
                s0_cart(i,j,2) = s0(j)
             end do
          end do

       else

          do j=lo(2),hi(2)
             do i=lo(1),hi(1)
                s0_cart(i,j,1) = s0(j)
             end do
          end do

       end if

    end if

  end subroutine put_1d_array_on_cart_2d

  subroutine put_1d_array_on_cart_3d(n,is_edge_centered,is_vector,s0,s0_cart,lo,hi,ng)

    use bl_constants_module
    use geometry, only: dr

    integer        , intent(in   ) :: n
    integer        , intent(in   ) :: lo(:),hi(:),ng
    logical        , intent(in   ) :: is_edge_centered,is_vector
    real(kind=dp_t), intent(in   ) :: s0(0:)
    real(kind=dp_t), intent(inout) :: s0_cart(lo(1)-ng:,lo(2)-ng:,lo(3)-ng:,:)

    integer :: i,j,k

    s0_cart = ZERO

    if (is_edge_centered) then

       if (is_vector) then

          do k=lo(3),hi(3)
             do j=lo(2),hi(2)
                do i=lo(1),hi(1)
                   s0_cart(i,j,k,3) = HALF * (s0(k) + s0(k+1))
                end do
             end do
          end do
          
       else
          
          do k=lo(3),hi(3)
             do j=lo(2),hi(2)
                do i=lo(1),hi(1)
                   s0_cart(i,j,k,1) = HALF * (s0(k) + s0(k+1))
                end do
             end do
          end do
          
       end if

    else

       if (is_vector) then

          do k=lo(3),hi(3)
             do j=lo(2),hi(2)
                do i=lo(1),hi(1)
                   s0_cart(i,j,k,3) = s0(k)
                end do
             end do
          end do
          
       else
          
          do k=lo(3),hi(3)
             do j=lo(2),hi(2)
                do i=lo(1),hi(1)
                   s0_cart(i,j,k,1) = s0(k)
                end do
             end do
          end do
          
       end if

    end if

  end subroutine put_1d_array_on_cart_3d

  subroutine put_1d_array_on_cart_3d_sphr(n,is_edge_centered,is_vector,s0,s0_cart,lo,hi, &
                                          dx,ng,normal)

    use bl_constants_module
    use geometry, only: dr, center, r_cc_loc, r_end_coord
    use probin_module, only: interp_type_radial_bin_to_cart

    integer        , intent(in   ) :: n
    integer        , intent(in   ) :: lo(:),hi(:),ng
    logical        , intent(in   ) :: is_edge_centered,is_vector
    real(kind=dp_t), intent(in   ) :: s0(0:)
    real(kind=dp_t), intent(inout) :: s0_cart(lo(1)-ng:,lo(2)-ng:,lo(3)-ng:,:)
    real(kind=dp_t), intent(in   ) :: dx(:)
    real(kind=dp_t), intent(in   ), optional :: normal(lo(1)-1:,lo(2)-1:,lo(3)-1:,:)

    integer         :: i,j,k,index
    real(kind=dp_t) :: x,y,z
    real(kind=dp_t) :: radius,rfac,s0_cart_val

    if (is_vector .and. (.not. present(normal)) ) then
       call bl_error('Error: Calling put_1d_array_on_cart_3d_sphr with is_vector=T and without normal')
    end if

    if (is_edge_centered) then

       ! interpolate from radial bin edge values

       do k = lo(3),hi(3)
          z = (dble(k)+HALF)*dx(3) - center(3)
          do j = lo(2),hi(2)
             y = (dble(j)+HALF)*dx(2) - center(2)
             do i = lo(1),hi(1)
                x = (dble(i)+HALF)*dx(1) - center(1)
                radius = sqrt(x**2 + y**2 + z**2)
                index  = int(radius / dr(n))
                
                rfac = (radius - dble(index)*dr(n)) / dr(n)
                s0_cart_val      = rfac * s0(index) + (ONE-rfac) * s0(index+1)

                if (is_vector) then
                   s0_cart(i,j,k,1) = s0_cart_val * normal(i,j,k,1)
                   s0_cart(i,j,k,2) = s0_cart_val * normal(i,j,k,2)
                   s0_cart(i,j,k,3) = s0_cart_val * normal(i,j,k,3)
                else
                   s0_cart(i,j,k,1) = s0_cart_val
                end if
             end do
          end do
       end do

    else

       ! interpolate from radial bin centered values

       do k = lo(3),hi(3)
          z = (dble(k)+HALF)*dx(3) - center(3)
          do j = lo(2),hi(2)
             y = (dble(j)+HALF)*dx(2) - center(2)
             do i = lo(1),hi(1)
                x = (dble(i)+HALF)*dx(1) - center(1)
                radius = sqrt(x**2 + y**2 + z**2)
                index  = int(radius / dr(n))
                
                if (interp_type_radial_bin_to_cart .eq. 1) then

                   s0_cart_val = s0(index)

                else if (interp_type_radial_bin_to_cart .eq. 2) then

                   if (radius .ge. r_cc_loc(n,index)) then
                      if (index .eq. r_end_coord(n)) then
                         s0_cart_val = s0(index)
                      else
                         s0_cart_val = s0(index+1)*(radius-r_cc_loc(n,index))/dr(n) &
                              + s0(index)*(r_cc_loc(n,index+1)-radius)/dr(n)
                      endif
                   else
                      if (index .eq. 0) then
                         s0_cart_val = s0(index)
                      else
                         s0_cart_val = s0(index)*(radius-r_cc_loc(n,index-1))/dr(n) &
                              + s0(index-1)*(r_cc_loc(n,index)-radius)/dr(n)
                      end if
                   end if

                else
                   call bl_error('Error: interp_type_radial_bin_to_cart not defined')
                end if

                if (is_vector) then
                   s0_cart(i,j,k,1) = s0_cart_val * normal(i,j,k,1)
                   s0_cart(i,j,k,2) = s0_cart_val * normal(i,j,k,2)
                   s0_cart(i,j,k,3) = s0_cart_val * normal(i,j,k,3)
                else
                   s0_cart(i,j,k,1) = s0_cart_val
                end if
             end do
          end do
       end do

    end if

  end subroutine put_1d_array_on_cart_3d_sphr

  subroutine put_1d_array_on_edges(n,s0,s0_cartx,s0_carty,s0_cartz,lo,hi,dx,ng)

    use bl_constants_module
    use geometry, only: dr, center, r_cc_loc, r_end_coord
    use probin_module, only: interp_type_radial_bin_to_cart

    integer        , intent(in   ) :: n
    integer        , intent(in   ) :: lo(:),hi(:),ng
    real(kind=dp_t), intent(in   ) :: s0(0:)
    real(kind=dp_t), intent(inout) :: s0_cartx(lo(1)-ng:,lo(2)-ng:,lo(3)-ng:)
    real(kind=dp_t), intent(inout) :: s0_carty(lo(1)-ng:,lo(2)-ng:,lo(3)-ng:)
    real(kind=dp_t), intent(inout) :: s0_cartz(lo(1)-ng:,lo(2)-ng:,lo(3)-ng:)
    real(kind=dp_t), intent(in   ) :: dx(:)

    integer         :: i,j,k,index
    real(kind=dp_t) :: x,y,z
    real(kind=dp_t) :: radius,rfac,s0_cart_val

    ! interpolate from radial bin centered values onto x-edges

    do k = lo(3),hi(3)
       z = (dble(k)+HALF)*dx(3) - center(3)
       do j = lo(2),hi(2)
          y = (dble(j)+HALF)*dx(2) - center(2)
          do i = lo(1),hi(1)+1
             x = (dble(i)     )*dx(1) - center(1)
             radius = sqrt(x**2 + y**2 + z**2)
             index  = int(radius / dr(n))
             
             if (interp_type_radial_bin_to_cart .eq. 1) then
                
                s0_cart_val = s0(index)
                
             else if (interp_type_radial_bin_to_cart .eq. 2) then
                
                if (radius .ge. r_cc_loc(n,index)) then
                   if (index .eq. r_end_coord(n)) then
                      s0_cart_val = s0(index)
                   else
                      s0_cart_val = s0(index+1)*(radius-r_cc_loc(n,index)  )/dr(n) &
                           + s0(index  )*(r_cc_loc(n,index+1)-radius)/dr(n)
                   endif
                else
                   if (index .eq. 0) then
                      s0_cart_val = s0(index)
                   else
                      s0_cart_val = s0(index  )*(radius-r_cc_loc(n,index-1))/dr(n) &
                           + s0(index-1)*(r_cc_loc(n,index)-radius)  /dr(n)
                   end if
                end if
                
             else
                call bl_error('Error: interp_type_radial_bin_to_cart not defined')
             end if
             
             s0_cartx(i,j,k) = s0_cart_val
          end do
       end do
    end do
    
    do k = lo(3),hi(3)
       z = (dble(k)+HALF)*dx(3) - center(3)
       do j = lo(2),hi(2)+1
          y = (dble(j)     )*dx(2) - center(2)
          do i = lo(1),hi(1)+1
             x = (dble(i)+HALF)*dx(1) - center(1)
             radius = sqrt(x**2 + y**2 + z**2)
             index  = int(radius / dr(n))
             
             if (interp_type_radial_bin_to_cart .eq. 1) then
                
                s0_cart_val = s0(index)
                
             else if (interp_type_radial_bin_to_cart .eq. 2) then
                
                if (radius .ge. r_cc_loc(n,index)) then
                   if (index .eq. r_end_coord(n)) then
                      s0_cart_val = s0(index)
                   else
                      s0_cart_val = s0(index+1)*(radius-r_cc_loc(n,index)  )/dr(n) &
                           + s0(index  )*(r_cc_loc(n,index+1)-radius)/dr(n)
                   endif
                else
                   if (index .eq. 0) then
                      s0_cart_val = s0(index)
                   else
                      s0_cart_val = s0(index  )*(radius-r_cc_loc(n,index-1))/dr(n) &
                           + s0(index-1)*(r_cc_loc(n,index)-radius  )/dr(n)
                   end if
                end if
                
             else
                call bl_error('Error: interp_type_radial_bin_to_cart not defined')
             end if
             
             s0_carty(i,j,k) = s0_cart_val
          end do
       end do
    end do
    
    do k = lo(3),hi(3)+1
       z = (dble(k)     )*dx(3) - center(3)
       do j = lo(2),hi(2)
          y = (dble(j)+HALF)*dx(2) - center(2)
          do i = lo(1),hi(1)+1
             x = (dble(i)+HALF)*dx(1) - center(1)
             radius = sqrt(x**2 + y**2 + z**2)
             index  = int(radius / dr(n))
             
             if (interp_type_radial_bin_to_cart .eq. 1) then
                
                s0_cart_val = s0(index)
                
             else if (interp_type_radial_bin_to_cart .eq. 2) then
                
                if (radius .ge. r_cc_loc(n,index)) then
                   if (index .eq. r_end_coord(n)) then
                      s0_cart_val = s0(index)
                   else
                      s0_cart_val = s0(index+1)*(radius-r_cc_loc(n,index)  )/dr(n) &
                           + s0(index  )*(r_cc_loc(n,index+1)-radius)/dr(n)
                   endif
                else
                   if (index .eq. 0) then
                      s0_cart_val = s0(index)
                   else
                      s0_cart_val = s0(index  )*(radius-r_cc_loc(n,index-1))/dr(n) &
                           + s0(index-1)*(r_cc_loc(n,index)-radius  )/dr(n)
                   end if
                end if
                
             else
                call bl_error('Error: interp_type_radial_bin_to_cart not defined')
             end if
             
             s0_cartz(i,j,k) = s0_cart_val
          end do
       end do
    end do

  end subroutine put_1d_array_on_edges

  subroutine make_3d_normal(normal,lo,hi,dx,ng)

    use bl_constants_module
    use geometry, only: spherical, center
    
    integer        , intent(in   ) :: lo(:),hi(:),ng
    real(kind=dp_t), intent(in   ) :: dx(:)
    real(kind=dp_t), intent(  out) :: normal(lo(1)-ng:,lo(2)-ng:,lo(3)-ng:,:)

    integer         :: i,j,k
    real(kind=dp_t) :: x,y,z,radius

    if (spherical .eq. 1) then
      do k = lo(3)-ng,hi(3)+ng
        z = (dble(k)+HALF)*dx(3) - center(3)
        do j = lo(2)-ng,hi(2)+ng
          y = (dble(j)+HALF)*dx(2) - center(2)
          do i = lo(1)-ng,hi(1)+ng
            x = (dble(i)+HALF)*dx(1) - center(1)
  
            radius = sqrt(x**2 + y**2 + z**2)
  
            normal(i,j,k,1) = x / radius
            normal(i,j,k,2) = y / radius
            normal(i,j,k,3) = z / radius
  
          end do
        end do
      end do
    else 
      call bl_error('SHOULDNT CALL MAKE_3D_NORMAL WITH SPHERICAL = 0')
    end if

  end subroutine make_3d_normal

end module fill_3d_module
