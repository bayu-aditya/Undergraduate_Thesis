module mod_hamiltonian
    double precision, allocatable, dimension(:,:) :: mat

    contains
    subroutine hamiltonian_fortran(k, num_orbitals, hamiltonian)
        implicit none
        double precision, dimension(3), intent(in) :: k
        integer, intent(in) :: num_orbitals
        double complex, dimension(num_orbitals, num_orbitals), intent(out) :: hamiltonian

        integer :: i, j, idx
        double precision, dimension(3) :: R

        hamiltonian = 0.0d0
        do idx = 1,ubound(mat,1)
            i = int(mat(idx,1))
            j = int(mat(idx,2))
            R(1) = mat(idx,5)
            R(2) = mat(idx,6)
            R(3) = mat(idx,7)
            hamiltonian(i,j) = hamiltonian(i,j) + ((mat(idx,3) + mat(idx,4)*cmplx(0.0,1.0))*exp(-cmplx(0.0,1.0)*dot_product(k,R)))
        end do
    end subroutine hamiltonian_fortran
end module mod_hamiltonian