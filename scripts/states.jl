using Distributed
addprocs(48; exeflags=["--project"])
@everywhere using RenyiEntropyOfMagic

prepare_states(20, 0.1:0.1:2.5)
