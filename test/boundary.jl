using OMEinsum
using LinearAlgebra

function boundary_vector(A, O)
    @assert size(A, 2)^4 == size(O, 1)
    X = kron(A, A, A, A)
    return vec(X * O * X')
end

A = rand(3, 2)
O = rand(2^4, 2^4)
V = ein"ab,cd,ef,gh,ij,kl,mn,op,jlnpbdfh->acegikmo"(A,A,A,A,A,A,A,A,reshape(O, ntuple(_->2, 8)));
dot(V, V)


st = vec(transpose(A) * A)
psi = kron(st, st, st, st)
psi = vec(permutedims(reshape(psi, ntuple(_->2, 8)), (1,3,5,7,2,4,6,8)))
psi' * kron(O, O) * psi


X = kron(A, A, A, A)
V = X * O * X'
dot(V, V)
(X * O * transpose(X))[]
st
dot(A[:, 1], A[:, 1])
dot(A[:, 1], A[:, 2])
dot(A[:, 2], A[:, 1])
dot(A[:, 2], A[:, 2])
