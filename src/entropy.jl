module Consts

using LinearAlgebra

const I2 = [1 0;0 1]
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

    T = lM * O * transpose(uM)
    shape = ntuple(4) do idx
        isodd(idx) ? size(A, 1) : size(A, 2)
    end
    T2 = reshape(T, shape)
    T3 = permutedims(T2, (1,3,2,4))
    return reshape(T3, size(A, 1)^2, size(A, 2)^2)
end

function boundary_vector(A, O)
    T = conj(A) * O * transpose(A)
    return vec(T)
end

function expectation(As, Os)
    lA, rA = first(As), last(As)
    lO, rO = first(Os), last(Os)
    lT = boundary_vector(lA, lO)
    rT = boundary_vector(rA, rO)
    T = rT
    for idx in length(As)-1:-1:2
        A, O = As[idx], Os[idx]
        T = transfer_matrix(A, O) * T
    end
    return transpose(lT) * T
end

function sample_paulistring(L::Int)
    labels_a = rand(0:1, L)
    labels_b = rand(0:1, L)
    
    return map(zip(labels_a, labels_b)) do (a, b)
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
end

function renyi_entropy(As; nsamples::Int=10^5)
    L = length(As)
    C = 2^L # 4^L / 2^L
    estimate = ThreadsX.sum(1:nsamples) do _
        Os = sample_paulistring(L)
        expectation(As, Os)^4
    end / nsamples * C
    return -log(estimate)
end

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

function exact_renyi_entropy(As)
    expect = sum(expectation(As, Os)^4 for Os in enumerate_paulistring(length(As)))    
    return -log(expect / 2^length(As))
end