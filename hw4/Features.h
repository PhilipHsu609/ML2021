#ifndef FEATURES_H
#define FEATURES_H

#include <Eigen/Dense>
#include <numeric>
#include <vector>

int C(int n, int r);

Eigen::MatrixXd polynomialFeatures(const Eigen::MatrixXd &X, int degree);

#endif