using Test
using RenyiEntropyOfMagic
using OMEinsum

function naive_two_site(A1, A2, O1, O2)
    psi = vec(ein"ji,jk->ik"(A1, A2))
    return psi' * reshape(ein"ik,jl->ijkl"(O1, O2), 4, 4) * psi
end

function naive_three_site(A1, A2, A3, O1, O2, O3)
    psi = vec(ein"ai,abj,bk->ijk"(A1,A2,A3))
    O = reshape(ein"il,jm,kn->ijklmn"(O1,O2,O3), length(psi), length(psi))
    return psi' * O * psi
end

A1, A2 = rand(3, 2), rand(3, 2)
O1, O2 = rand(2, 2), rand(2, 2)


@test naive_two_site(A1, A2, O1, O2) ≈ boundary_vector(A1, O1)' * boundary_vector(A2, O2)


A1, A2, A3 = rand(3, 2), rand(3, 3, 2), rand(3, 2)
O1, O2, O3 = rand(2, 2), rand(2, 2), rand(2, 2)


result = boundary_vector(A1, O1)' * transfer_matrix(A2, O2) * boundary_vector(A3, O3)
target = naive_three_site(A1, A2, A3, O1, O2, O3)
@test result ≈ target


function enumerate_paulistring(L::Int)
    labels_a = Iterators.product([0:1 for _ in 1:L]...)
    labels_b = Iterators.product([0:1 for _ in 1:L]...)
    
    return map(Iterators.product(labels_a, labels_b)) do (as, bs)
        map(zip(as, bs)) do (a, b)
            if a == 0 && b == 0
                Consts.I2
            elseif a == 0 && b == 1
                Consts.Z
            elseif a == 1 && b == 0
                Consts.X
            else # 11
                Consts.X * Consts.Z
            end
        end
    end |> vec
end

As = rand(3, 2), rand(3, 3, 2), rand(3, 2)
sum(expectation(As, Os)^4 for Os in enumerate_paulistring(3))

using Statistics

mean(1:10^6) do _
    Os = sample_paulistring(3)
    expectation(As, Os)^4
end * 4^3
renyi_entropy(As)
