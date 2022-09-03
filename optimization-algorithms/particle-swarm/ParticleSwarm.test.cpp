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

#include <gtest/gtest.h>
#include "ParticleSwarm.hpp"

double Rosenbrock(std::array<double,2> u)
{
    double x = u[0];
    double y = u[1];

    double a = 1.0;
    double b = 10.0;
    return (a - x) * (a - x) + b * ((y - (x * x)) * (y - (x * x)));
}

double Himmelblau(std::array<double, 2> u)
{
    double x = u[0];
    double y = u[1];

    return (x * x + y - 11) * (x * x + y - 11) + (x + y * y - 7) * (x + y * y - 7);
}

double ThreeState(std::array<double, 3> u)
{
    double x = u[0];
    double y = u[1];
    double z = u[2];

    return (x * x + y - 11) * (x * x + y - 11) + (x + y * y - 7) * (x + y * y - 7) + (z - 1) * (z - 1);
}

TEST(ParticleSwarm, Rosenbrock_MaxIteration)
{
    ParticleSwarm<2, 10> pso(Rosenbrock);
    pso.config.maxIter = 10;
    pso.config.costThreshold = 0;
    pso.config.maxVel = 0.01;
    pso.config.varMin = -1.0;
    pso.config.varMax = 1.0;
    pso.config.print = true;
    std::array<double,2> result = pso.Solve();

    std::array<double, 2> expected{ 1.0, 1.0 };
    EXPECT_GE(expected, result);
}

TEST(ParticleSwarm, Rosenbrock_CostThreshold)
{
    ParticleSwarm<2, 10> pso(Rosenbrock);
    pso.config.maxIter = 1000;
    pso.config.costThreshold = 1e-5;
    pso.config.maxVel = 0.01;
    pso.config.varMin = -1.0;
    pso.config.varMax = 1.0;
    std::array<double, 2> result = pso.Solve();

    std::array<double, 2> expected{ 1.0, 1.0 };
    EXPECT_GE(expected, result);
}

TEST(ParticleSwarm, Rosenbrock_SwarmSize_Low)
{
    ParticleSwarm<2, 3> pso(Rosenbrock);
    pso.config.maxIter = 100;
    pso.config.costThreshold = 0;
    pso.config.maxVel = 0.01;
    std::array<double, 2> result = pso.Solve();

    std::array<double, 2> expected{ 1.0, 1.0 };
    EXPECT_GE(expected, result);
}

TEST(ParticleSwarm, Rosenbrock_SwarmSize_High)
{
    ParticleSwarm<2, 3000> pso(Rosenbrock);
    pso.config.maxIter = 1000;
    pso.config.costThreshold = 0;
    pso.config.maxVel = 0.01;
    std::array<double, 2> result = pso.Solve();

    std::array<double, 2> expected{ 1.0, 1.0 };
    EXPECT_EQ(expected, result);
}

TEST(ParticleSwarm, Rosenbrock_MaxVelocity)
{
    ParticleSwarm<2, 10> pso1(Rosenbrock);
    pso1.config.maxIter = 1000;
    pso1.config.costThreshold = 0;
    pso1.config.maxVel = 0.0001;
    std::array<double, 2> result1 = pso1.Solve();

    ParticleSwarm<2, 10> pso2(Rosenbrock);
    pso2.config.maxIter = 1000;
    pso2.config.costThreshold = 0;
    pso2.config.maxVel = 0.1;
    std::array<double, 2> result2 = pso2.Solve();

    EXPECT_GE(result2, result1);
}

TEST(ParticleSwarm, Rosenbrock_Print)
{
    ParticleSwarm<2, 100> pso(Rosenbrock);
    pso.config.maxIter = 1000;
    pso.config.costThreshold = 0;
    pso.config.maxVel = 0.1;
    pso.config.print = true;
    std::array<double, 2> result = pso.Solve();

    std::array<double, 2> expected{ 1.0, 1.0 };
    EXPECT_EQ(expected, result);
}

TEST(ParticleSwarm, Himmelblau_Minimas)
{
    ParticleSwarm<2, 10> pso(Himmelblau);
    pso.config.maxIter = 1000;
    pso.config.costThreshold = 0;
    pso.config.varMax = 5.0;
    pso.config.varMin = -5.0;
    std::array<double, 2> result = pso.Solve();

    std::array<double, 2> expected1{ -3.779310, -3.283186 };
    std::array<double, 2> expected2{ -2.805118, 3.131312 };
    std::array<double, 2> expected3{ 3.0, 2.0 };
    std::array<double, 2> expected4{ 3.584428, -1.848126 };
    double threshold = 1e-6;
    double x = result[0];
    if (x > 3.0)
    {
        EXPECT_NEAR(expected4[0], result[0], threshold);
        EXPECT_NEAR(expected4[1], result[1], threshold);
    }
    else if (x > 0.0)
    {
        EXPECT_NEAR(expected3[0], result[0], threshold);
        EXPECT_NEAR(expected3[1], result[1], threshold);
    }
    else if (x > -3.0)
    {
        EXPECT_NEAR(expected2[0], result[0], threshold);
        EXPECT_NEAR(expected2[1], result[1], threshold);
    }
    else
    {
        EXPECT_NEAR(expected1[0], result[0], threshold);
        EXPECT_NEAR(expected1[1], result[1], threshold);
    }
}

TEST(ParticleSwarm, ThreeStateTest)
{
    ParticleSwarm<3, 10> pso(ThreeState);
    pso.config.maxIter = 10000;
    pso.config.costThreshold = 0;
    pso.config.varMax = 10.0;
    pso.config.varMin = -10.0;
    pso.config.maxVel = 0.1;
    pso.config.print = false;
    std::array<double, 3> result = pso.Solve();

    std::array<double, 3> expected1{ -3.779310, -3.283186, 1.0 };
    std::array<double, 3> expected2{ -2.805118, 3.131312, 1.0 };
    std::array<double, 3> expected3{ 3.0, 2.0, 1.0 };
    std::array<double, 3> expected4{ 3.584428, -1.848126, 1.0 };
    double threshold = 1e-6;
    double x = result[0];
    if (x > 3.0)
    {
        EXPECT_NEAR(expected4[0], result[0], threshold);
        EXPECT_NEAR(expected4[1], result[1], threshold);
        EXPECT_NEAR(expected4[2], result[2], threshold);
    }
    else if (x > 0.0)
    {
        EXPECT_NEAR(expected3[0], result[0], threshold);
        EXPECT_NEAR(expected3[1], result[1], threshold);
        EXPECT_NEAR(expected3[2], result[2], threshold);
    }
    else if (x > -3.0)
    {
        EXPECT_NEAR(expected2[0], result[0], threshold);
        EXPECT_NEAR(expected2[1], result[1], threshold);
        EXPECT_NEAR(expected2[2], result[2], threshold);
    }
    else
    {
        EXPECT_NEAR(expected1[0], result[0], threshold);
        EXPECT_NEAR(expected1[1], result[1], threshold);
        EXPECT_NEAR(expected1[2], result[2], threshold);
    }
}