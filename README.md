## Optimization Algorithms

[![Build Status](https://github.com/onurae/optimization-algorithms/actions/workflows/ci.yml/badge.svg)](https://github.com/onurae/optimization-algorithms/actions/workflows/ci.yml)&nbsp;&nbsp;
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/onurae/optimization-algorithms/blob/main/LICENSE)

A library of derivative-free optimization algorithms.
- Golden Section
- Adaptive Nelder-Mead
- Particle Swarm

### Golden Section

It is a technique for finding a minimum of a one dimensional function inside a specified interval.
[Read more](https://en.wikipedia.org/wiki/Golden-section_search)

#### Example

One dimensional objective function:

```cpp
double Func(double x)
{
    return (x - 2) * (x * x - 6);
}
```

To find the minimum value of the objective function in the range [0, 4]

```cpp
double xBound1 = 0.0;
double xBound2 = 4.0;
double tolerance = 1e-4;
double xResult = GoldenSectionSearch(Func, xBound1, xBound2, tolerance);
```

### Adaptive Nelder-Mead

It is a heuristic search method used to find the minimum of an objective function in a multidimensional space. 
[Read more](https://en.wikipedia.org/wiki/Nelder%E2%80%93Mead_method)

#### Reference
Fuchang Gao and Lixing Han, (2012)<br>
Implementing the Nelder-Mead simplex algorithm with adaptive parameters.<br>
Computational Optimization and Applications, 51, (1), 259-277.<br>

#### Example

Two dimensional [Himmelblau](https://en.wikipedia.org/wiki/Himmelblau%27s_function) objective function.
It has four identical local minima.

```cpp
double Himmelblau(std::array<double, 2> u) {
    double x = u[0];
    double y = u[1];

    return (x * x + y - 11) * (x * x + y - 11) + (x + y * y - 7) * (x + y * y - 7);
}
```

Create a two dimensional Nelder-Mead method.
```cpp
NelderMead<2> nm(Himmelblau);
```
Set configuration parameters.
- Length of edges of the initial simplex
- Maximum iteration number
- Termination tolerance on f(x)
- Termination tolerance on x

```cpp
nm.config.edge = 0.1;  // Length of edges of the initial simplex.
nm.config.maxIter = 0; // Zero by default. It means no iteration limit.
nm.config.fTol = 0.0;  // f(x) tolerance.
nm.config.xTol = 1e-6; // x tolerance.
```

### Particle Swarm

It is an algorithm for finding optimal regions of complex search spaces through the interaction of individuals in a population of particles.
[Read more](https://en.wikipedia.org/wiki/Particle_swarm_optimization)

#### Reference
Clerc, M. and Kennedy, J. (2002)<br>
The Particle Swarm: Explosion, Stability, and Convergence in a Multi-Dimensional Complex Space.<br>
IEEE Transactions on Evolutionary Computation, 6, 58-73.<br>

#### Example

Two dimensional [Rosenbrock](https://en.wikipedia.org/wiki/Rosenbrock_function) objective function.
The minimum is at [1,1] for this example.
```cpp
double Rosenbrock(std::array<double,2> u) {
    double x = u[0];
    double y = u[1];
    double a = 1.0;
    double b = 100.0;
    return (a - x) * (a - x) + b * ((y - (x * x)) * (y - (x * x)));
}
```

Create a two dimensional swarm of 20 particles.
```cpp
ParticleSwarm<2, 20> pso(Rosenbrock);
```
Set configuration parameters.
- Maximum iteration number
- Cost threshold to stop iteration
- Maximum absolute particle velocity
- Lower and upper bounds of variables
```cpp
pso.config.maxIter = 1000;
pso.config.costThreshold = 1e-5;
pso.config.maxVel = 0.1;
pso.config.varMin = -2.0;
pso.config.varMax = 2.0;
```
Turn on the print option if needed.

```cpp
pso.config.print = true;
```

Start the optimization.
```cpp
std::array<double, 2> result = pso.Solve();
```
