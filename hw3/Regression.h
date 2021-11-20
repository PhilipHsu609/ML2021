#ifndef REGRESSION_H
#define REGRESSION_H

#include <Eigen/Dense>
#include <Eigen/QR>
#include <chrono>
#include <functional>
#include <iostream>
#include <random>

class ErrorMeasure {
   public:
    ErrorMeasure() = default;
    static double squareError(Eigen::VectorXd &&y_pred, Eigen::VectorXd &y_true);
    static double zeroOneError(Eigen::VectorXd &&y_pred, Eigen::VectorXd &y_true);
};

class LinearRegression {
   public:
    LinearRegression() = default;
    void fit(Eigen::MatrixXd &X, Eigen::VectorXd &y);
    Eigen::VectorXd predict(Eigen::MatrixXd &X);

   private:
    Eigen::VectorXd w;  // weights
};

#endif