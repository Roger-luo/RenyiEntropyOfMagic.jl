using Comonicon

@cast function scan_tfim(N::Int; nsamples::Int=10^6, nprocs::Int=0)
    nprocs > 0 && addprocs(nprocs; exeflags=["--project=$(pkgdir(RenyiEntropyOfMagic))"])
    @everywhere using RenyiEntropyOfMagic
    results = scan_tfim_renyi_entropy(N; nsamples)
    serialize(pkgdir(RenyiEntropyOfMagic, "data", "$N-sites-result.dat"), results)
end

@cast function prepare_state(N::Int;nprocs::Int=0)
    nprocs > 0 && addprocs(nprocs; exeflags=["--project=$(pkgdir(RenyiEntropyOfMagic))"])
    @everywhere using RenyiEntropyOfMagic
    prepare_states(N, 0.1:0.1:2.5)
end

@main