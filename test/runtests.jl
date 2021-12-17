using RenyiEntropyOfMagic
using OMEinsum
using SparseArrays
using LinearAlgebra
using Test

function naive_two_site(st)
    psi = reshape(kron(st, st, st, st), ntuple(_->2, 8))
    O = kron(Consts.O, Consts.O)
    P = ein"abcdefgh,acegbdfhijklmnop->ijklmnop"(psi,reshape(O, ntuple(_->2, 16)))
    return ein"abcdefgh,acegbdfh->"(psi, P)[]
end

function naive_three_site(st)
    psi = reshape(kron(st, st, st, st), ntuple(_->2, 12))
    O = kron(Consts.O, Consts.O, Consts.O)
    P = ein"abcdefghijkl,adgjbehkcfilmnopqrstuvwx->mnopqrstuvwx"(psi,reshape(O, ntuple(_->2, 24)))
    return ein"abcdefghijkl,adgjbehkcfil->"(psi, P)[]
end

@testset "random periodic boundary MPS" begin
    # random MPS
    A = rand(3, 3, 2)
    T = transfer_matrix(A)

    # tests
    # periodic boundary
    # 1 site
    st = ein"iij->j"(A)
    psi = kron(st, st, st, st)
    psi' * Consts.O * psi
    tr(T)
    @test psi' * Consts.O * psi ≈ tr(T)

    # 2 site
    st = vec(ein"ija,jib->ab"(A, A))
    @test naive_two_site(st) ≈ tr(T^2)

    A1 = rand(2, 3, 2)
    A2 = rand(3, 2, 2)
    T1 = transfer_matrix(A1)
    T2 = transfer_matrix(A2)
    st = vec(ein"ija,jib->ab"(A1, A2))
    @test naive_two_site(st) ≈ tr(T1 * T2)
end

@testset "random open boundary MPS" begin
    # 2 site
    lA, rA = rand(3, 2), rand(3, 2)
    lT, rT = boundary_vector(lA), boundary_vector(rA)

    st = vec(transpose(lA) * rA)
    expect = naive_two_site(st)
    @test expect ≈ dot(lT, rT)
    @test expect ≈ replicated_expectation([lA, rA])

    # 3 site
    lA, A, rA = rand(3, 2), rand(3, 3, 2), rand(3, 2)
    lT, T, rT = boundary_vector(lA), transfer_matrix(A), boundary_vector(rA)

    st = vec(ein"ia,ijb,jc->abc"(lA, A, rA))
    expect = naive_three_site(st)
    @test expect ≈ dot(lT, T * rT)
    @test expect ≈ replicated_expectation([lA, A, rA])
end
