/******************************************************************************************
*                                                                                         *
*    Adaptive Nelder-Mead Simplex Method [Test]                                           *
*    It is a heuristic search method used to find the minimum                             *
*    of an objective function in a multidimensional space.                                *
*                                                                                         *
*    Copyright (c) 2022 Onur AKIN <https://github.com/onurae>                             *
*    Licensed under the MIT License.                                                      *
*                                                                                         *
*    Fuchang Gao and Lixing Han, (2012)                                                   *
*    Implementing the Nelder-Mead simplex algorithm with adaptive parameters.             *
*    Computational Optimization and Applications, 51, (1), 259-277.                       *
*                                                                                         *
******************************************************************************************/

#include <gtest/gtest.h>
#include "NelderMead.hpp"

double Rosenbrock(std::array<double, 2> u) {
    double x = u[0];
    double y = u[1];

    double alpha = 10.0;
    return (1 - x)*(1 - x) + alpha * ((y - (x*x)) *(y - (x*x)));
}

double Himmelblau(std::array<double, 2> u) {
    double x = u[0];
    double y = u[1];

    return (x*x + y - 11) * (x*x + y - 11) + (x + y * y - 7) * (x + y * y - 7);
}

double ThreeState(std::array<double, 3> u) {
    double x = u[0];
    double y = u[1];
    double z = u[2];

    return (x*x + y - 11) * (x*x + y - 11) + (x + y * y - 7) * (x + y * y - 7) + (z-1)*(z-1);
}

TEST(NelderMead, Rosenbrock_Exact)
{
    NelderMead<2> nm(Rosenbrock);
    nm.config.xTol = 0.0;
    std::array<double, 2> xStart = { -1.0, 1.0 };
    auto res = nm.Solve(xStart);

    std::array<double, 2> expected{ 1.0, 1.0 };
    EXPECT_EQ(expected, res);
}

TEST(NelderMead, Rosenbrock_Threshold_X)
{
    NelderMead<2> nm(Rosenbrock);
    nm.config.xTol = 1e-5;
    nm.config.print = true;
    std::array<double, 2> xStart = { -2.0, 2.0 };
    auto res = nm.Solve(xStart);
    
    std::array<double, 2> expected{ 1.0, 1.0 };
    EXPECT_NEAR(expected[0], res[0], nm.config.xTol);
    EXPECT_NEAR(expected[1], res[1], nm.config.xTol);
}

TEST(NelderMead, Rosenbrock_Threshold_F)
{
    NelderMead<2> nm(Rosenbrock);
    nm.config.xTol = 1.0;
    nm.config.fTol = 1e-5;
    std::array<double, 2> xStart = { -2.0, 2.0 };
    auto res = nm.Solve(xStart);

    double threshold = 1e-2;
    std::array<double, 2> expected{ 1.0, 1.0 };
    EXPECT_NEAR(expected[0], res[0], threshold);
    EXPECT_NEAR(expected[1], res[1], threshold);
}

TEST(NelderMead, Rosenbrock_MaxIteration)
{
    NelderMead<2> nm(Rosenbrock);
    nm.config.maxIter = 50;
    std::array<double, 2> xStart = { -1.0, 1.0 };
    auto res = nm.Solve(xStart);

    double threshold = 1e-3;
    std::array<double, 2> expected{ 1.0, 1.0 };
    EXPECT_NEAR(expected[0], res[0], threshold);
    EXPECT_NEAR(expected[1], res[1], threshold);
}

TEST(NelderMead, Rosenbrock_Print)
{
    NelderMead<2> nm(Rosenbrock);
    nm.config.maxIter = 50;
    nm.config.print = true;
    std::array<double, 2> xStart = { -1.0, 1.0 };
    auto res = nm.Solve(xStart);

    double threshold = 1e-3;
    std::array<double, 2> expected{ 1.0, 1.0 };
    EXPECT_NEAR(expected[0], res[0], threshold);
    EXPECT_NEAR(expected[1], res[1], threshold);
}

TEST(NelderMead, Rosenbrock_Exact_Edge)
{
    NelderMead<2> nm(Rosenbrock);
    nm.config.xTol = 0.0;
    nm.config.edge = 1e-4;
    std::array<double, 2> xStart = { -1.0, 1.0 };
    auto res = nm.Solve(xStart);

    std::array<double, 2> expected{ 1.0, 1.0 };
    EXPECT_EQ(expected, res);
}

TEST(NelderMead, Rosenbrock_EdgeLowerThanTolerance)
{
    NelderMead<2> nm(Rosenbrock);
    nm.config.xTol = 1e-5;
    nm.config.edge = 1e-6;
    std::array<double, 2> xStart = { -1.0, 1.0 };
    auto res = nm.Solve(xStart);

    std::array<double, 2> expected{ -1.0, 1.0 };
    EXPECT_EQ(expected, res);
}

TEST(NelderMead, Rosenbrock_DefaultConfig)
{
    NelderMead<2> nm(Rosenbrock);
    std::array<double, 2> xStart = { -1.0, 1.0 };
    auto res = nm.Solve(xStart);

    double threshold = 1e-4;
    std::array<double, 2> expected{ 1.0, 1.0 };
    EXPECT_NEAR(expected[0], res[0], threshold);
    EXPECT_NEAR(expected[1], res[1], threshold);
}

TEST(NelderMead, Himmelblau_Minima1)
{
    NelderMead<2> nm(Himmelblau);
    nm.config.xTol = 1e-6;
    nm.config.edge = 1e-3;
    std::array<double, 2> xStart = { -6.0, -6.0 };
    auto res = nm.Solve(xStart);

    std::array<double, 2> expected{ -3.779310, -3.283186 };
    EXPECT_NEAR(expected[0], res[0], nm.config.xTol);
    EXPECT_NEAR(expected[1], res[1], nm.config.xTol);
}

TEST(NelderMead, Himmelblau_Minima2)
{
    NelderMead<2> nm(Himmelblau);
    nm.config.xTol = 1e-6;
    nm.config.edge = 1e-3;
    std::array<double, 2> xStart = { -6.0, 6.0 };
    auto res = nm.Solve(xStart);

    std::array<double, 2> expected{ -2.805118, 3.131312 };
    EXPECT_NEAR(expected[0], res[0], nm.config.xTol);
    EXPECT_NEAR(expected[1], res[1], nm.config.xTol);
}

TEST(NelderMead, Himmelblau_Minima3)
{
    NelderMead<2> nm(Himmelblau);
    nm.config.xTol = 1e-6;
    nm.config.edge = 1e-3;
    std::array<double, 2> xStart = { 6.0, 6.0 };
    auto res = nm.Solve(xStart);

    std::array<double, 2> expected{ 3.0, 2.0 };
    EXPECT_NEAR(expected[0], res[0], nm.config.xTol);
    EXPECT_NEAR(expected[1], res[1], nm.config.xTol);
}

TEST(NelderMead, Himmelblau_Minima4)
{
    NelderMead<2> nm(Himmelblau);
    nm.config.xTol = 1e-6;
    nm.config.edge = 1e-3;
    std::array<double, 2> xStart = { 6.0, -6.0 };
    auto res = nm.Solve(xStart);

    std::array<double, 2> expected{ 3.584428, -1.848126 };
    EXPECT_NEAR(expected[0], res[0], nm.config.xTol);
    EXPECT_NEAR(expected[1], res[1], nm.config.xTol);
}

TEST(NelderMead, Himmelblau_HugeTriangle)
{
    NelderMead<2> nm(Himmelblau);
    nm.config.xTol = 1e-6;
    nm.config.edge = 12.0;
    std::array<double, 2> xStart = { -12.0, -12.0 };
    auto res = nm.Solve(xStart);

    std::array<double, 2> expected{ 3.0, 2.0 };
    EXPECT_NEAR(expected[0], res[0], nm.config.xTol);
    EXPECT_NEAR(expected[1], res[1], nm.config.xTol);
}

TEST(NelderMead, Himmelblau_TinyTriangle)
{
    NelderMead<2> nm(Himmelblau);
    nm.config.xTol = 1e-8;
    nm.config.edge = 1e-5;
    std::array<double, 2> xStart = { 3.0, 3.0 };
    auto res = nm.Solve(xStart);

    std::array<double, 2> expected{ 3.0, 2.0 };
    EXPECT_NEAR(expected[0], res[0], nm.config.xTol);
    EXPECT_NEAR(expected[1], res[1], nm.config.xTol);
}

TEST(NelderMead, ThreeStateTest)
{
    NelderMead<3> nm(ThreeState);
    nm.config.xTol = 1e-6;
    nm.config.edge = 1e-3;
    std::array<double, 3> xStart = { -6.0, -6.0, -6.0 };
    auto res = nm.Solve(xStart);

    std::array<double, 3> expected{ -3.779310, -3.283186, 1.0 };
    EXPECT_NEAR(expected[0], res[0], nm.config.xTol);
    EXPECT_NEAR(expected[1], res[1], nm.config.xTol);
    EXPECT_NEAR(expected[2], res[2], nm.config.xTol);
}