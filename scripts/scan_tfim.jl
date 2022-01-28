using Distributed
addprocs(48; exeflags=["--project"])
@everywhere begin
    using Serialization
    using RenyiEntropyOfMagic
end

N = parse(Int, ARGS[1])
results = scan_tfim_renyi_entropy(N; nsamples=10^6)
serialize(pkgdir(RenyiEntropyOfMagic, "data", "$N-sites-result.dat"), results)
