/******************************************************************************************
*                                                                                         *
*    Adaptive Nelder-Mead Simplex Method                                                  *
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

#ifndef NELDERMEAD_HPP
#define NELDERMEAD_HPP

#include <array>
#include <algorithm>
#include <functional>

/*!
Nelder Mead.
\tparam dim Number of dimensions.
*/
template<int dim>
class NelderMead
{
private:
    std::function<double(std::array<double, dim>)> func;    //!< Cost function.

    struct Config
    {
        bool print{ false };        //!< Print switch.
        double edge{ 0.1 };         //!< Edges of the initial simplex.
        int maxIter{ 0 };           //!< Maximum number of iterations.
        double fTol{ 1e-4 };        //!< Termination tolerance on f(x).
        double xTol{ 1e-4 };        //!< Termination tolerance on x.
    };

    double alpha{ 1.0 };                //!< Reflection parameter.
    double gamma{ 1.0 + 2.0 / dim };    //!< Expansion parameter.
    double rho{ 0.75 - 0.5 / dim };     //!< Contraction parameter.
    double sigma{ 1.0 - 1.0 / dim };    //!< Shrink parameter.

    struct Point
    {
        std::array<double, dim> position = {};
        double cost;
    };
public:
    explicit NelderMead(const std::function<double(std::array<double, dim>)>& func);
    virtual ~NelderMead() = default;
    std::array<double, dim> Solve(const std::array<double, dim>& xStart);
    Config config;
};

/*!
\tparam dim Number of dimensions.
\param func Cost function.
*/
template<int dim>
inline NelderMead<dim>::NelderMead(const std::function<double(std::array<double, dim>)>& func) :
    func(func)
{
}

/*!
Find the best cost position.
\tparam dim Number of dimensions.
\param xStart Initial position.
\return std::array<double, dim> The best cost position.
*/
template<int dim>
inline std::array<double, dim> NelderMead<dim>::Solve(const std::array<double, dim>& xStart)
{
    std::vector<Point> points;
    points.push_back(Point{ xStart, func(xStart) });

    for (int i = 0; i < dim; i++)
    {
        auto x(xStart);
        x[i] += config.edge;
        points.push_back(Point{ x, func(x) });
    }

    int iteration = 0;
    bool fTolFlag = false;
    bool xTolFlag = false;
    while (true)
    {
        // Iteration information
        if (config.print == true)
        {
            std::cout << "[" << iteration << "] " << "f(x): " << points[0].cost << " | ";
            for (int i = 0; i < dim; i++)
            {
                std::cout << "x" << i + 1 << ": " << points[0].position[i];
                if (i != (dim - 1))
                {
                    std::cout << ", ";
                }
            }
            std::cout << std::endl;
        }

        // Termination
        if (config.maxIter && iteration >= config.maxIter)
        {
            if (config.print == true)
            {
                std::cout << "Done." << std::endl << "Max.Iteration: " << std::to_string(iteration) << std::endl;
            }
            return points[0].position;
        }

        fTolFlag = true;
        for (int k = 1; k < points.size(); k++)
        {
            if (std::abs(points[k].cost - points[0].cost) > config.fTol)
            {
                fTolFlag = false;
                break;
            }
        }
        xTolFlag = true;
        for (int k = 1; k < points.size(); k++)
        {
            for (int i = 0; i < dim; i++)
            {
                if (std::abs(points[k].position[i] - points[0].position[i]) > config.xTol)
                {
                    xTolFlag = false;
                    k = dim + 1;
                    break;
                }
            }
        }
        if (fTolFlag && xTolFlag)
        {
            if (config.print == true)
            {
                std::cout << "Done." << std::endl << "Iteration: " << std::to_string(iteration) << std::endl;
            }
            return points[0].position;
        }
        iteration++;

        // Order
        std::sort(points.begin(), points.end(), [](const Point& p1, const Point& p2) -> bool { return (p1.cost < p2.cost); });

        // Centroid
        std::array<double, dim> centroidPosition = {};
        for (int k = 0; k < dim; k++)
        {
            for (int i = 0; i < dim; i++)
            {
                centroidPosition[i] += points[k].position[i];
            }
        }
        for (int i = 0; i < dim; i++)
        {
            centroidPosition[i] /= dim;
        }

        // Reflection
        Point reflectionPoint;
        for (int i = 0; i < dim; i++)
        {
            reflectionPoint.position[i] = centroidPosition[i] + alpha * (centroidPosition[i] - points.back().position[i]);
        }
        reflectionPoint.cost = func(reflectionPoint.position);
        if ((points[0].cost <= reflectionPoint.cost) && (reflectionPoint.cost < points[points.size() - 2].cost))
        {
            points.pop_back();
            points.push_back(Point{ reflectionPoint.position, reflectionPoint.cost });
            continue;
        }

        // Expansion
        if (reflectionPoint.cost < points[0].cost)
        {
            Point expansionPoint;
            for (int i = 0; i < dim; i++)
            {
                expansionPoint.position[i] = centroidPosition[i] + gamma * (reflectionPoint.position[i] - centroidPosition[i]);
            }
            expansionPoint.cost = func(expansionPoint.position);
            if (expansionPoint.cost < reflectionPoint.cost)
            {
                points.pop_back();
                points.push_back(Point{ expansionPoint.position, expansionPoint.cost });
                continue;
            }
            points.pop_back();
            points.push_back(Point{ reflectionPoint.position, reflectionPoint.cost });
            continue;
        }

        // Contraction
        Point contractionPoint;
        if (reflectionPoint.cost < points.back().cost)
        {
            // Outside
            for (int i = 0; i < dim; i++)
            {
                contractionPoint.position[i] = centroidPosition[i] + rho * (reflectionPoint.position[i] - centroidPosition[i]);
            }
            contractionPoint.cost = func(contractionPoint.position);
            if (contractionPoint.cost < reflectionPoint.cost)
            {
                points.pop_back();
                points.push_back(Point{ contractionPoint.position, contractionPoint.cost });
                continue;
            }
        }
        else 
        {
            // Inside
            for (int i = 0; i < dim; i++)
            {
                contractionPoint.position[i] = centroidPosition[i] + rho * (points.back().position[i] - centroidPosition[i]);
            }
            contractionPoint.cost = func(contractionPoint.position);
            if (contractionPoint.cost < points.back().cost)
            {
                points.pop_back();
                points.push_back(Point{ contractionPoint.position, contractionPoint.cost });
                continue;
            }
        }

        // Shrink
        std::vector<Point> newPoints;
        for (int k = 1; k < points.size(); k++)
        {
            Point newPoint;
            for (int i = 0; i < dim; i++)
            {
                newPoint.position[i] = points[0].position[i] + sigma * (points[k].position[i] - points[0].position[i]);
            }
            newPoint.cost = func(newPoint.position);
            newPoints.push_back(Point{ newPoint.position, newPoint.cost });
        }
        points.resize(1);
        points.insert(points.end(), newPoints.begin(), newPoints.end());
    }
}

#endif /* NELDERMEAD_HPP */