#ifndef FEATURES_H
#define FEATURES_H

#include <Eigen/Dense>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

int C(int n, int r);

Eigen::MatrixXd polynomialFeatures(const Eigen::MatrixXd &X, int degree);

Eigen::MatrixXd homogeneousPolynomialFeatures(const Eigen::MatrixXd &X, int degree);

Eigen::MatrixXd lessFeatures(const Eigen::MatrixXd &X, int n);

Eigen::MatrixXd randomFeatures(const Eigen::MatrixXd &X, int n, int seed = 0);

#endif