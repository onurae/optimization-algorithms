/******************************************************************************************
*                                                                                         *
*    Golden Section Search                                                                *
*    It is a technique for finding a minimum of a function inside a specified interval.   *
*                                                                                         *
*    Copyright (c) 2022 Onur AKIN <https://github.com/onurae>                             *
*    Licensed under the MIT License.                                                      *
*                                                                                         *
******************************************************************************************/

#ifndef GOLDENSECTION_HPP
#define GOLDENSECTION_HPP

#include <cmath>
#include <iostream>

/*!
Golden section search.
\param func Cost function.
\param xB1 Bound1.
\param xB2 Bound2.
\param tolerance Tolerance.
\param print Print switch.
\return xF The final value.
*/
template<typename T>
double GoldenSectionSearch(const T& func, double xB1, double xB2, double tolerance, bool print = false)
{
    double r = (std::sqrt(5.0) - 1.0) * 0.5;

    double xL = std::min(xB1, xB2);
    double xU = std::max(xB1, xB2);

    double xL0 = xL;
    double xU0 = xU;
    double fL0 = func(xL);
    double fU0 = func(xU);
    
    double x1 = xL + r * (xU - xL);
    double x2 = xU - r * (xU - xL);
    double f1 = func(x1);
    double f2 = func(x2);

    while ((xU - xL) > tolerance)
    {
        if (f1 < f2)
        {
            xL = x2;
            x2 = x1;
            f2 = f1;
            x1 = xL + r * (xU - xL);
            f1 = func(x1);
        }
        else
        {
            xU = x1;
            x1 = x2;
            f1 = f2;
            x2 = xU - r * (xU - xL);
            f2 = func(x2);
        }
    }
    double fL = func(xL);
    double fU = func(xU);

    double xF;
    double fF;
    if (fL < fU)
    {
        xF = xL;
        fF = fL;
    }
    else
    {
        xF = xU;
        fF = fU;
    }

    if (fL0 < fF)
    {
        xF = xL0;
        fF = fL0;
    }
    if (fU0 < fF)
    {
        xF = xU0;
        fF = fU0;
    }

    if (print == true)
    {
        std::cout << std::setprecision(15) << "x: " << xF << "\t" << "y: " << fF << std::endl;
    }

    return xF;
};

#endif /* GOLDENSECTION_HPP */
