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
