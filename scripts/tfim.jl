# In this example we show how to pass a DMRGObserver to
# the dmrg function which allows tracking energy convergence and
# convergence of local operators.
using ITensors
using RenyiEntropyOfMagic
using TerminalLoggers
using Logging: global_logger
using TerminalLoggers: TerminalLogger

if !@isdefined(VSCodeServer)
    global_logger(TerminalLogger())
end

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
length(As)
ranks=Ranks(
    left_boundary=16,
    right_boundary=16,
    middle=[(16, 16) for _ in 1:length(As)-2]
)

A1, A2, A3, A4, A5 = As



replicated_expectation(As; ranks)
replicated_expectation(As)
renyi_entropy_of_magic(As)
renyi_entropy_of_magic(As; ranks)
# nsites = length(As)
# Os = [Consts.O for _ in 1:length(As)]
# lA, rA = first(As), last(As)
# lO, rO = first(Os), last(Os)

# lK, lU = boundary_decomposition(lA, lO; rank=ranks.left_boundary)
# rK, rU = boundary_decomposition(rA, rO; rank=ranks.right_boundary)

# K, prevU = rK, rU
# idx = length(As)-2
# A, O = As[idx], Os[idx]
# left, right = ranks.middle[idx-1]


# # C, U1, U2 = middle_decomposition(A, O; left, right)
# T = left_right_kron(A)
# C, U1, U2 = hosvd3(T; left, right)
# sandwitch(C)
# @time svd(reshape(T, size(T, 1), :));

# reshape(T, size(T, 1), :)

# M = transpose(U2) * prevU
# K = C * kron(M, M) * K
