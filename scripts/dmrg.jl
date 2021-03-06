# In this example we show how to pass a DMRGObserver to
# the dmrg function which allows tracking energy convergence and
# convergence of local operators.
using ITensors

"""
  Get MPO of transverse field Ising model Hamiltonian with field strength h
"""
function tfimMPO(sites, h::Float64)
  # Input operator terms which define a Hamiltonian
  N = length(sites)
  ampo = OpSum()
  for j in 1:(N - 1)
    ampo += -1, "Z", j, "Z", j + 1
  end
  for j in 1:N
    ampo += h, "X", j
  end
  # Convert these terms to an MPO tensor network
  return MPO(ampo, sites)
end

N = 100
sites = siteinds("S=1/2", N)
psi0 = randomMPS(sites; linkdims=16)

# define parameters for DMRG sweeps
sweeps = Sweeps(15)
setmaxdim!(sweeps, 10, 20, 100, 100, 200)
setcutoff!(sweeps, 1E-8)

H = tfimMPO(sites, 1.0)
energy, psi = dmrg(H, psi0, sweeps)


As = map(psi.data) do t
    T = reshape(t.tensor.storage.data, size(t))
    if ndims(T) == 2
      return T
    else
      return permutedims(T, (3, 1, 2))
    end
end

size.(As)
maximum(map(x->maximum(size(x)), As))
length(Ranks(As).middle)
Ranks(As)

using RenyiEntropyOfMagic

ranks=Ranks(
    left_boundary=8,
    right_boundary=8,
    middle=[(8, 8) for _ in 1:98]
)

replicated_expectation(As; ranks)


nsites = length(As)
Os = [Consts.O for _ in 1:length(As)]
lA, rA = first(As), last(As)
lO, rO = first(Os), last(Os)

lK, lU = boundary_decomposition(lA, lO; rank=ranks.left_boundary)
rK, rU = boundary_decomposition(rA, rO; rank=ranks.right_boundary)

K, prevU = rK, rU
idx = length(As)-1
A, O = As[idx], Os[idx]
left, right = ranks.middle[idx-1]
C, U1, U2 = middle_decomposition(A, O; left, right)

M = transpose(U2) * prevU
K = C * kron(M, M) * K
