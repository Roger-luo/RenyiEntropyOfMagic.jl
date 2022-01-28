export tfimMPO

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

export get_matrices
function get_matrices(N, h)
  sites = siteinds("S=1/2", N)
  psi0 = randomMPS(sites; linkdims=20)

  # define parameters for DMRG sweeps
  sweeps = Sweeps(50)
  setmaxdim!(sweeps, 10, 20, 100, 100, 200)
  setcutoff!(sweeps, 1E-10)

  H = tfimMPO(sites, h)
  energy, psi = dmrg(H, psi0, sweeps)
  @show energy

  return map(psi.data) do t
      T = reshape(t.tensor.storage.data, size(t))
      if ndims(T) == 2
        return T
      else
        return permutedims(T, (3, 1, 2))
      end
  end
end

export tfim_renyi_entropy
function tfim_renyi_entropy(N, h)
  As = get_matrices(N, h)
  return renyi_entropy(As; nsamples=10^6)/N
end

export exact_tfim_renyi_entropy
function exact_tfim_renyi_entropy(N, h)
  As = get_matrices(N, h)
  return exact_renyi_entropy(As)/N
end

data_dir(xs...) = joinpath(pkgdir(RenyiEntropyOfMagic, "data", xs...))

export prepare_states
function prepare_states(N, hs)
    isdir(data_dir()) || mkpath(data_dir())
    As = pmap(hs) do h
        get_matrices(N, h)
    end
    serialize(data_dir("$N-sites.mps"), hs=>As)
end

function read_matrices(N)
    deserialize(data_dir("$N-sites.mps"))
end

export scan_tfim_renyi_entropy
function scan_tfim_renyi_entropy(N; nsamples=10^6)
    hs, As = read_matrices(N)
    return pmap(zip(hs, As)) do (h, A)
        @info "running" h
        return renyi_entropy(A; nsamples)/N
    end
end
