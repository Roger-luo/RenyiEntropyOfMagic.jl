using Distributed
addprocs(5; exeflags=["--project"])
@everywhere using RenyiEntropyOfMagic
using Serialization

mc_M2 = pmap(0.1:0.1:2.5) do h
  @info "running" h
  tfim_renyi_entropy(20, h)
end
serialize("20.dat", mc_M2)


# using Yao
# L = 8
# tfim_h = -sum(kron(L, i=>Z, mod1(i+1, L)=>Z) for i in 1:L) - sum(put(L, i=>X) for i in 1:L)
# mat(Z)
# H = Matrix(mat(tfim_h))
# using LinearAlgebra

# eigmin(H)
# eigvals(H)
# psi = eigvecs(H)[:, 1]

# Os = enumerate_paulistring(L)

# -log(sum((psi' * kron(O...) * psi)^4 for O in Os)/2^L)/L

# exact_tfim_renyi_entropy(8, 1.0)

# As = get_matrices(8, 1.0)
# exact_renyi_entropy(As)

# renyi_entropy(As; nsamples=10^6)

# mc_M2 = [tfim_renyi_entropy(20, h) for h in 0.1:0.1:2.5]

# exact_M2 = [exact_tfim_renyi_entropy(290, h) for h in 0.1:0.1:2.5]

# tfim_renyi_entropy(20, 1.0)
# exact_tfim_renyi_entropy(8, 0.1)


# using CairoMakie

# lines(exact_M2)

# i = Index(2, "S=1/2")
# j = Index(2, "S=1/2")
# println(ITensors.op("X", i))