module RenyiEntropyOfMagic

using LinearAlgebra

export Consts, transfer_matrix, boundary_vector, replicated_expectation

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

function replicated_expectation(As, Os = [Consts.O for _ in 1:length(As)])
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

function renyi_entropy_of_magic(As, Os = [Consts.O for _ in 1:length(As)])
    return -log(2^length(As) * replicated_expectation(As, Os))
end

end
