function simple_tm(A)
    A_dag = conj(A)
    uM = reshape(A, :, size(A, 3))
    lM = reshape(A_dag, :, size(A, 3))
    uP = kron(uM, uM, uM, uM)
    lP = kron(lM, lM, lM, lM)

    T = lP * Consts.O * transpose(uP)
    shape = ntuple(8) do idx
        isodd(idx) ? size(A, 1) : size(A, 2)
    end
    T2 = reshape(T, shape)
    T3 = permutedims(T2, (1,3,5,79,11,13,15, 2,4,6,8,10,12,14,16))
    return reshape(T3, size(A, 1)^8, size(A, 2)^8)
end
