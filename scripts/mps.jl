using OMEinsum
A1, A2, A3 = rand(ComplexF64, 2, 3, 2), rand(ComplexF64, 3, 3, 2), rand(ComplexF64, 3, 2, 2)

#  ----------------------
# |                      |
# | i     j      k      i|
#  --[A1]---[A2]---[A3]--
#     |      |      |
#     a      b      c
st1 = vec(ein"ija,jkb,kic->abc"(A1, A2, A3))

B1,B2,B3 = conj(permutedims(A3, (2, 1, 3))), conj(permutedims(A2, (2, 1, 3))), conj(permutedims(A1, (2, 1, 3)))
st2 = vec(ein"ija,jkb,kic->cba"(B1, B2, B3))
size(B3)
size(A1)

C1,C2,C3 = conj(A1), conj(A2), conj(A3)

st3 = vec(ein"ija,jkb,kic->abc"(C1, C2, C3))



T1 = reshape(ein"ija,pqa->ipjq"(A1,C1), 4, 9)
T2 = reshape(ein"ija,pqa->ipjq"(A2,C2), 9, 9)
T3 = reshape(ein"ija,pqa->ipjq"(A3,C3), 9, 4)

ein"abab->"(reshape(T1 * T2 * T3, 2, 2, 2, 2))

vec(st1')

sum(abs2.(st1))

transpose(st2) * st1

reshape(A1, :, 2) * transpose(reshape(B1, :, 2))


T1 = reshape(A1, :, 2) * transpose(reshape(conj(A1), :, 2))
T2 = reshape(A2, :, 2) * transpose(reshape(conj(A2), :, 2))
T3 = reshape(A3, :, 2) * transpose(reshape(conj(A3), :, 2))

M1 = permutedims(reshape(T1, 2, 3, 3, 2), (1, 4, 2, 3))
M2 = permutedims(reshape(T2, 3, 3, 3, 3), (1, 4, 2, 3))
M3 = permutedims(reshape(T2, 3, 2, 2, 3), (1, 4, 2, 3))