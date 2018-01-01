# Reversible jump MCMC

This is an implementation of a reversible jump MCMC algorithm.
The program first generates a set of samples from a known mixture of gaussians, and then tries to infer a mixture of gaussians from the data using the reversible jump algorithm.

Starting from a one-component mixture centered at the origin, each iteration mutates the current mixture and jumps across mixture models by adding (splitting) or removing (merging) components.
This implementation is based on Robert & Casella's excellent book "Monte Carlo Statistical Methods", and more particularly the chapter on reversible jump algorithms (p.425).

## Building the program

    g++ -std=c++11 -O3 -DNDEBUG -march=native rev_jump.cpp -o rev_jump
    
## License

MIT
