using Test
using RenyiEntropyOfMagic
using LinearAlgebra
using OMEinsum

A = rand(5, 2)
L, U = boundary_decomposition(A)
@test kron(U, U) * L ≈ boundary_vector(A)

L, U = boundary_decomposition(A; rank=8)
@test size(L) == (64, )
@test size(U) == (625, 8)

T = rand(3, 5, 2)
C, U1, U2 = hosvd3(T)
@test ein"ijk,pi,qj->pqk"(C, U1, U2) ≈ T

# TODO: construct a test core tensor
# C, U1, U2 = hosvd3(T; left=3, right=4)
# norm(ein"ijk,pi,qj->pqk"(C, U1, U2) - T)

A = rand(2, 3, 2)
T = left_right_kron(A)
L = ein"abc,def,ghi,jkl->adgjbehkcfil"(A, A, A, A);
L = reshape(L, (size(A, 1)^4, size(A, 2)^4, 16))
@test L ≈ T

@test transfer_matrix(A) ≈ sandwitch(T)

C, U1, U2 = hosvd3(T)
T3 = sandwitch(C)
@test kron(U1, U1) * T3 * kron(transpose(U2), transpose(U2)) ≈ transfer_matrix(A)

T, U1, U2 = middle_decomposition(A)
@test kron(U1, U1) * T * kron(transpose(U2), transpose(U2)) ≈ transfer_matrix(A)


A1, A2, A3 = rand(3, 2), rand(3, 3, 2), rand(3, 2)
T1, T2, T3 = boundary_vector(A1), transfer_matrix(A2), boundary_vector(A3)

K1, U1 = boundary_decomposition(A1)
K2, U2, U3 = middle_decomposition(A2)
K3, U4 = boundary_decomposition(A3)

@test kron(U1, U1) * K1 ≈ T1
@test kron(U2, U2) * K2 * kron(transpose(U3), transpose(U3)) ≈ T2
@test kron(U4, U4) * K3 ≈ T3

M1 = transpose(U1) * U2
M2 = transpose(U3) * U4

@test transpose(K1) * kron(M1, M1) * K2 * kron(M2, M2) * K3 ≈ dot(T1, T2 * T3)

replicated_expectation([A1, A2, A3]) ≈ dot(T1, T2 * T3)


A1, A2, A3, A4 = rand(2, 2), rand(2, 3, 2), rand(3, 2, 2), rand(2, 2)
T1, T2, T3, T4 = boundary_vector(A1), transfer_matrix(A2), transfer_matrix(A3), boundary_vector(A4)

K1, U1 = boundary_decomposition(A1)
K2, U2, U3 = middle_decomposition(A2)
K3, U4, U5 = middle_decomposition(A3)
K4, U6 = boundary_decomposition(A4)

M1 = transpose(U1) * U2
M2 = transpose(U3) * U4
M3 = transpose(U5) * U6

size(U6)
size(U5)
@test transpose(K1) * kron(M1, M1) * K2 * kron(M2, M2) * K3 * kron(M3, M3) * K4
    ≈ transpose(T1) * T2 * T3 * T4

@test replicated_expectation([A1, A2, A3, A4]) ≈ transpose(T1) * T2 * T3 * T4


Ranks([A1, A2, A3, A4])


ranks=Ranks(
    left_boundary=8,
    right_boundary=8,
    middle=[
        (8, 16),
        (16, 8),
    ]
)
replicated_expectation([A1, A2, A3, A4]; ranks)
replicated_expectation([A1, A2, A3, A4])
