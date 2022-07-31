/******************************************************************************************
*                                                                                         *
*    Golden Section Search [Test]                                                         *
*    It is a technique for finding a minimum of a function inside a specified interval.   *
*                                                                                         *
*    Copyright (c) 2022 Onur AKIN <https://github.com/onurae>                             *
*    Licensed under the MIT License.                                                      *
*                                                                                         *
******************************************************************************************/

#include <gtest/gtest.h>
#include "GoldenSection.hpp"

double Equation1(double x)
{
    return ((x - 2.0) * (x - 2.0));
}

double Equation2(double x)
{
    return (2.0 * std::sin(x) - (x * x / 10.0));
}

double Equation3(double x)
{
    if (x < 5)
    {
        return ((x - 2.0) * (x - 2.0));
    }
    return ((x - 8.0) * (x - 8.0));
}

double Equation4(double x)
{
    if (x < 0.2)
    {
        return -4.0;
    }
    if (x > 3.8)
    {
        return -5.0;
    }
    return ((x - 2.0) * (x - 2.0));
}

double Equation5(double x)
{
    if (x < 0.2)
    {
        return -5.0;
    }
    return ((x - 2.0) * (x - 2.0));
}

TEST(GoldenSectionSearch, SeachingForMinima)
{
    double xL = -10.0;
    double xU = 10.0;
    double tolerance = 1e-5;
    double result = GoldenSectionSearch(Equation1, xL, xU, tolerance, true);
    double expected = 2.0;
    EXPECT_NEAR(expected, result, tolerance);
}

TEST(GoldenSectionSearch, SolutionIsAtTheUpperBound)
{
    double xL = 0.0;
    double xU = 4.0;
    double tolerance = 1e-5;
    double result = GoldenSectionSearch(Equation2, xL, xU, tolerance, true);
    double expected = 4.0;
    EXPECT_EQ(expected, result);
}

TEST(GoldenSectionSearch, SolutionIsAtTheLowerBound)
{
    double xL = 0.0;
    double xU = 2.0;
    double tolerance = 1e-5;
    double result = GoldenSectionSearch(Equation2, xL, xU, tolerance, true);
    double expected = 0.0;
    EXPECT_EQ(expected, result);
}

TEST(GoldenSectionSearch, MultipleMinima1)
{
    double xL = 0.0;
    double xU = 9.0;
    double tolerance = 1e-5;
    double result = GoldenSectionSearch(Equation3, xL, xU, tolerance, true);
    double expected = 2.0;
    EXPECT_NEAR(expected, result, tolerance);
}

TEST(GoldenSectionSearch, MultipleMinima2)
{
    double xL = 1.0;
    double xU = 10.0;
    double tolerance = 1e-5;
    double result = GoldenSectionSearch(Equation3, xL, xU, tolerance, true);
    double expected = 8.0;
    EXPECT_NEAR(expected, result, tolerance);
}

TEST(GoldenSectionSearch, ConvergedToLocalMinimaButSolutionIsAtTheUpperBound)
{
    double xL = 0.0;
    double xU = 4.0;
    double tolerance = 1e-5;
    double result = GoldenSectionSearch(Equation4, xL, xU, tolerance, true);
    double expected = 4.0;
    EXPECT_EQ(expected, result);
}

TEST(GoldenSectionSearch, ConvergedToLocalMinimaButSolutionIsAtTheLowerBound)
{
    double xL = 0.0;
    double xU = 4.0;
    double tolerance = 1e-5;
    double result = GoldenSectionSearch(Equation5, xL, xU, tolerance, true);
    double expected = 0.0;
    EXPECT_EQ(expected, result);
}

TEST(GoldenSectionSearch, BoundReverseInput)
{
    double xL = 0.0;
    double xU = 4.0;
    double tolerance = 1e-5;
    double result = GoldenSectionSearch(Equation1, xU, xL, tolerance, true);
    double expected = 2.0;
    EXPECT_NEAR(expected, result, tolerance);
}

TEST(GoldenSectionSearch, HighTolerance)
{
    double xL = 1.8;
    double xU = 2.4;
    double tolerance = 1.0;
    double result = GoldenSectionSearch(Equation1, xL, xU, tolerance, true);
    double expected = 1.8;
    EXPECT_NEAR(expected, result, tolerance);
}
