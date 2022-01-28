using Distributed
addprocs(48; exeflags=["--project"])
@everywhere begin
    using Serialization
    using RenyiEntropyOfMagic
end

results = scan_tfim_renyi_entropy(20; nsamples=10^4)
serialize(pkgdir(RenyiEntropyOfMagic, "data", "20-sites-result.dat"), results)
