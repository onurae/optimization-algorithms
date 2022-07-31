/********************************************************************************************************
*                                                                                                       *
*    Particle Swarm Optimization                                                                        *
*    It is an algorithm for finding optimal regions of complex search spaces through the interaction    *
*    of individuals in a population of particles.                                                       *
*                                                                                                       *
*    Copyright (c) 2022 Onur AKIN <https://github.com/onurae>                                           *
*    Licensed under the MIT License.                                                                    *
*                                                                                                       *
*    Clerc, M. and Kennedy, J. (2002)                                                                   *
*    The Particle Swarm: Explosion, Stability, and Convergence in a Multi-Dimensional Complex Space.    *
*    IEEE Transactions on Evolutionary Computation, 6, 58-73.                                           *
*                                                                                                       *
********************************************************************************************************/

#ifndef PARTICLESWARM_HPP
#define PARTICLESWARM_HPP

#include <array>
#include <functional>
#include <random>

/*!
Particle swarm.
\tparam dim Number of dimensions.
\tparam nSwarm Swarm size.
*/
template<int dim, int nSwarm>
class ParticleSwarm
{
private:
    std::function<double(std::array<double, dim>)> func;    //!< Cost function.

    struct Config
    {
        int maxIter{ 200 * dim };                   //!< Break after maxIter iterations.
        double costThreshold{ 0.0 };                //!< Cost threshold.
        double varMin{ -1.0 };                      //!< Lower bound of variables.
        double varMax{ 1.0 };                       //!< Upper bound of variables.
        double maxVel{ 0.1 * (varMax - varMin) };   //!< Maximum absolute velocity.
        bool print{ false };                        //!< Print switch.
    };

    struct Particle
    {
        std::array<double, dim> position;       //!< Particle position.
        std::array<double, dim> velocity;       //!< Particle velocity.
        double cost;                            //!< Particle cost.
        double bestCost;                        //!< Particle best cost.
        std::array<double, dim> bestPosition;   //!< Particle best cost position.
    };

    double GetR();
public:
    explicit ParticleSwarm(const std::function<double(std::array<double, dim>)>& func);
    virtual ~ParticleSwarm() = default;
    std::array<double, dim> Solve();
    Config config;
};

/*!
\tparam dim Number of dimensions.
\tparam nSwarm Swarm size.
\param func Cost function.
*/
template<int dim, int nSwarm>
inline ParticleSwarm<dim, nSwarm>::ParticleSwarm(const std::function<double(std::array<double, dim>)>& func) :
    func(func)
{
}

/*!
Find the best cost position.
\tparam dim Number of dimensions.
\tparam nSwarm Swarm size.
\return std::array<double, dim> Global best cost position.
*/
template<int dim, int nSwarm>
inline std::array<double, dim> ParticleSwarm<dim, nSwarm>::Solve()
{
    std::array<Particle, nSwarm> particles;
    double globalBestCost{ std::numeric_limits<double>::max() };    //!< Global best cost.
    std::array<double, dim> globalBestPosition;                     //!< Global best cost position.

    // Uniform random distribution
    std::random_device rd;  // Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
    std::array<std::uniform_real_distribution<>, dim> disArray;
    for (int d = 0; d < dim; d++)
    {
        std::uniform_real_distribution<> distr(config.varMin, config.varMax); // [varMin, varMax)
        disArray[d] = distr;
    }

    for (int i = 0; i < nSwarm; i++)
    {
        // Create a particle
        particles[i] = Particle();

        // Initialize position
        for (int k = 0; k < dim; k++)
        {
            particles[i].position[k] = disArray[k](gen);
        }

        // Initialize velocity
        particles[i].velocity.fill(0);

        // Calculate cost
        particles[i].cost = func(particles[i].position);

        // Update the personal best
        particles[i].bestPosition = particles[i].position;
        particles[i].bestCost = particles[i].cost;

        // Update the global best
        if (particles[i].cost < globalBestCost)
        {
            globalBestCost = particles[i].bestCost;
            globalBestPosition = particles[i].bestPosition;
        }
    }

    // Constriction coefficients
    double kappa = 1.0;
    double phi1 = 2.05;
    double phi2 = 2.05;
    double phi = phi1 + phi2;
    double chi = 2.0 * kappa / std::abs(2.0 - phi - std::sqrt(phi * phi - 4.0 * phi));
    double w = chi;             //!< Inertia coefficient.
    double c1 = chi * phi1;     //!< Self acceleration coefficient.
    double c2 = chi * phi2;     //!< Social acceleration coefficient.

    int iteration = 0;
    while (true)
    {
        // Iteration information
        if (config.print == true)
        {
            std::cout << "[" << iteration << "] " << "f(x): " << globalBestCost << " | ";
            for (int i = 0; i < dim; i++)
            {
                std::cout << "x" << i + 1 << ": " << globalBestPosition[i];
                if (i != (dim - 1))
                {
                    std::cout << ", ";
                }
            }
            std::cout << std::endl;
        }

        // Return when maxIter is reached.
        if (iteration >= config.maxIter)
        {
            if (config.print == true)
            {
                std::cout << "Done." << " [Max.Iteration]" << std::endl;
            }
            return globalBestPosition;
        }

        // Return when global best cost is under or equal to the cost threshold.
        if (globalBestCost <= config.costThreshold)
        {
            if (config.print == true)
            {
                std::cout << "Done." << " [Cost Threshold] " << "[" << globalBestCost << " <= " << config.costThreshold << "]" << std::endl;
            }
            return globalBestPosition;
        }

        iteration++;

        for (int i = 0; i < nSwarm; i++)
        {
            // Update velocity
            for (int k = 0; k < dim; k++)
            {
                particles[i].velocity[k] = w * particles[i].velocity[k]                             // Inertia term.
                    + c1 * GetR() * (particles[i].bestPosition[k] - particles[i].position[k])       // Cognitive term.
                    + c2 * GetR() * (globalBestPosition[k] - particles[i].position[k]);             // Social term.
            }

            // Limit velocity
            for (int k = 0; k < dim; k++)
            {
                particles[i].velocity[k] = std::max(particles[i].velocity[k], -config.maxVel);
                particles[i].velocity[k] = std::min(particles[i].velocity[k], config.maxVel);
            }

            // Update position
            for (int k = 0; k < dim; k++)
            {
                particles[i].position[k] = particles[i].position[k] + particles[i].velocity[k];
            }

            // Upper-Lower bound limits
            for (int k = 0; k < dim; k++)
            {
                particles[i].position[k] = std::max(particles[i].position[k], config.varMin);
                particles[i].position[k] = std::min(particles[i].position[k], config.varMax);
            }

            // Calculate cost
            particles[i].cost = func(particles[i].position);

            // Update the particle best
            if (particles[i].cost < particles[i].bestCost)
            {
                particles[i].bestCost = particles[i].cost;
                particles[i].bestPosition = particles[i].position;

                // Update the global best
                if (particles[i].cost < globalBestCost)
                {
                    globalBestCost = particles[i].bestCost;
                    globalBestPosition = particles[i].bestPosition;
                }
            }
        }
    }
}

/*!
Returns a random value [0,1].
\tparam dim Number of dimensions.
\tparam nSwarm Swarm size.
*/
template<int dim, int nSwarm>
inline double ParticleSwarm<dim, nSwarm>::GetR()
{
    return (std::rand() / double(RAND_MAX));
}

#endif /* PARTICLESWARM_HPP */