module RenyiEntropyOfMagic

using LinearAlgebra

export Consts, Ranks, transfer_matrix, boundary_vector,
    hosvd3, sandwitch, left_right_kron,
    boundary_decomposition,
    middle_decomposition,
    replicated_expectation,
    renyi_entropy_of_magic,
    exact_replicated_expectation,
    exact_renyi_entropy_of_magic

module Consts

using LinearAlgebra

const X = [0 1;1 0]
const Z = [1 0;0 -1]
const G_X = 0.5 * (kron(X, X, X, X) + I(2^4))
const G_Z = 0.5 * (kron(Z, Z, Z, Z) + I(2^4))
const O = G_X * G_Z

end

function transfer_matrix(A, O = Consts.O)
    A_dag = conj(A)
    uM = reshape(A, :, size(A, 3))
    lM = reshape(A_dag, :, size(A, 3))
    uP = kron(uM, uM, uM, uM)
    lP = kron(lM, lM, lM, lM)

    T = lP * O * transpose(uP)
    shape = ntuple(16) do idx
        isodd(idx) ? size(A, 1) : size(A, 2)
    end
    T2 = reshape(T, shape)
    T3 = permutedims(T2, (1,3,5,7,9,11,13,15, 2,4,6,8,10,12,14,16))
    return reshape(T3, size(A, 1)^8, size(A, 2)^8)
end

function boundary_vector(A, O = Consts.O)
    A_dag = conj(A)
    uP = kron(A, A, A, A)
    lP = kron(A_dag, A_dag, A_dag, A_dag)
    T = lP * O * transpose(uP)
    return vec(T)
end

"""
specialized decomposition of the following tensor

```
            -[T]-  --->  -[U1]--[C]--[U2]-
              |                  |
```
"""
function hosvd3(T; left=size(T, 1), right=size(T, 2))
    M1 = reshape(T, size(T, 1), :)
    U1, _, _ = svd(M1)
    T′ = permutedims(T, (2, 1, 3))
    U2, _, _ = svd(reshape(T′, size(T, 2), :))

    if left < size(T, 1)
        U1 = U1[:, 1:left]
    end

    if right < size(T, 2)
        U2 = U2[:, 1:right]
    end

    D1_shape = (left, size(T, 2), size(T, 3))
    D2_shape = (right, left, size(T, 3))

    # ijk,pi,qj->pqk(T, U1', U2')
    D1 = U1' * M1 # pjk
    D1′ = permutedims(reshape(D1, D1_shape), (2, 1, 3))
    D2 = U2' * reshape(D1′, size(T, 2), :)
    C = permutedims(reshape(D2, D2_shape), (2, 1, 3))
    return C, U1, U2
end

function left_right_kron(A::AbstractArray{<:Any, 3})
    M = reshape(A, :, size(A, 3))
    P = kron(M, M, M, M)
    shape = ntuple(8) do idx
        isodd(idx) ? size(A, 1) : size(A, 2)
    end
    T = reshape(P, (shape..., 16));
    T = permutedims(T, (1, 3, 5, 7, 2, 4, 6, 8, 9));
    return reshape(T, (size(A, 1)^4, size(A, 2)^4, 16))
end

function sandwitch(T, O = Consts.O)
    P = reshape(T, :, size(T, 3))
    P = conj(P) * O * transpose(P)
    P = reshape(P, size(T, 1), size(T, 2), size(T, 1), size(T, 2))
    P = permutedims(P, (1, 3, 2, 4))
    return reshape(P, size(T, 1)^2, size(T, 2)^2)
end

function middle_decomposition(A, O = Consts.O; left=size(A, 1)^4, right=size(A, 2)^4)
    T = left_right_kron(A)
    C, U1, U2 = hosvd3(T; left, right)
    return sandwitch(C), U1, U2
end

function boundary_decomposition(A, O = Consts.O; rank=size(A, 1)^4)
    P = kron(A, A, A, A)
    U, S, V = svd(P)
    if rank < size(P, 1)
        U = U[:, 1:rank]
    end

    C = U' * P
    T = conj(C) * O * transpose(C)
    return vec(T), U
end

Base.@kwdef struct Ranks
    left_boundary::Int
    right_boundary::Int
    middle::Vector{Tuple{Int, Int}}
end

function Ranks(As)
    ranks = Tuple{Int, Int}[]
    for idx in 2:length(As)-1
        A = As[idx]
        l_rank = min(size(A, 1), size(A, 2) * size(A, 3))
        r_rank = min(size(A, 2), size(A, 1) * size(A, 3))
        push!(ranks, (l_rank^4, r_rank^4))
    end

    lA, rA = first(As), last(As)
    return Ranks(minimum(size(lA))^4, minimum(size(rA))^4, ranks)
end

function replicated_expectation(As, Os = [Consts.O for _ in 1:length(As)]; ranks::Ranks=Ranks(As))
    nsites = length(As)
    lA, rA = first(As), last(As)
    lO, rO = first(Os), last(Os)

    lK, lU = boundary_decomposition(lA, lO; rank=ranks.left_boundary)
    rK, rU = boundary_decomposition(rA, rO; rank=ranks.right_boundary)

    K, prevU = rK, rU
    for idx in length(As)-1:-1:2
        @show idx
        A, O = As[idx], Os[idx]
        left, right = ranks.middle[idx-1]
        C, U1, U2 = middle_decomposition(A, O; left, right)

        M = transpose(U2) * prevU
        K = C * kron(M, M) * K
        prevU = U1
    end
    M = transpose(lU) * prevU
    return transpose(lK) * kron(M, M) * K
end

function renyi_entropy_of_magic(As, Os = [Consts.O for _ in 1:length(As)]; ranks::Ranks=Ranks(As))
    return -log(2^length(As) * replicated_expectation(As, Os; ranks))
end

function exact_replicated_expectation(As, Os = [Consts.O for _ in 1:length(As)])
    nsites = length(As)
    lA, rA = first(As), last(As)
    lO, rO = first(Os), last(Os)

    lM = boundary_vector(lA, lO)
    rM = boundary_vector(rA, rO)

    M = rM
    for idx in 2:length(As)-1
        A, O = As[idx], Os[idx]
        M = transfer_matrix(A, O) * M
    end
    return dot(lM, M)
end

function exact_renyi_entropy_of_magic(As, Os = [Consts.O for _ in 1:length(As)])
    return -log(2^length(As) * exact_replicated_expectation(As, Os))
end

end
