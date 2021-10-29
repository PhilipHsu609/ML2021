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
    double squareError(Eigen::VectorXd &y_pred, Eigen::VectorXd &y_true);
    double zeroOneError(Eigen::VectorXd &y_pred, Eigen::VectorXd &y_true);
};

class Dataset {
   public:
    Dataset();
    void samples(int trainSize = 200, int testSize = 5000);
    void setSeed(int seed = 0);
    void addOutliers(int size = 20);

    Eigen::MatrixXd X_train, X_test;  // training set
    Eigen::VectorXd y_train, y_test;  // testing set

   private:
    int flipCoin();                       // Return -1 or 1
    Eigen::Vector3d singleSample(int y);  // single sample [1, x1, x2]
    std::mt19937 mersenne;
};

class LinearRegression : public ErrorMeasure {
   public:
    LinearRegression() = default;
    void fit(Eigen::MatrixXd &X, Eigen::VectorXd &y);
    Eigen::VectorXd predict(Eigen::MatrixXd &X);

   private:
    Eigen::VectorXd w;  // weights
};

class LogisticRegression : public ErrorMeasure {
   public:
    LogisticRegression(double eta, int T);
    void fit(Eigen::MatrixXd &X, Eigen::VectorXd &y);
    Eigen::VectorXd predict(Eigen::MatrixXd &X);

   private:
    double eta;         // learning rate
    int T;              // iterations
    Eigen::VectorXd w;  // weights
};

#endif