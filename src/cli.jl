using Comonicon

@cast function scan_tfim(N::Int; lognsamples::Int=6, nprocs::Int=1, nthreads::Int=2)
    if nprocs > 1
        procs = addprocs(nprocs-1; exeflags=["--project=$(pkgdir(RenyiEntropyOfMagic))", "--threads=$nthreads"])
        Distributed.remotecall_eval(Main, procs, Expr(:toplevel, :(using RenyiEntropyOfMagic)))
    end
    results = scan_tfim_renyi_entropy(N; nsamples=10^lognsamples)
    serialize(pkgdir(RenyiEntropyOfMagic, "data", "$N-sites-result.dat"), results)
end

@cast function prepare_state(N::Int;nprocs::Int=1)
    if nprocs > 1
        procs = addprocs(nprocs-1; exeflags=["--project=$(pkgdir(RenyiEntropyOfMagic))"])
        Distributed.remotecall_eval(Main, procs, Expr(:toplevel, :(using RenyiEntropyOfMagic)))
    end
    prepare_states(N, 0.1:0.1:2.5)
end

@main