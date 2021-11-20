#include "Regression.h"

void LinearRegression::fit(Eigen::MatrixXd &X, Eigen::VectorXd &y) {
    Eigen::MatrixXd pinv = X.completeOrthogonalDecomposition().pseudoInverse();
    w = pinv * y;
}

Eigen::VectorXd LinearRegression::predict(Eigen::MatrixXd &X) {
    return X * w;
}

double ErrorMeasure::squareError(Eigen::VectorXd &&y_pred, Eigen::VectorXd &y_true) {
    double N = y_pred.rows();
    return (y_pred - y_true).squaredNorm() / N;
}

double ErrorMeasure::zeroOneError(Eigen::VectorXd &&y_pred, Eigen::VectorXd &y_true) {
    double N = y_pred.rows();

    std::function<double(double)> sign = [](double val) -> double {
        return val >= 0 ? 1.0 : -1.0;
    };

    Eigen::VectorXd diff = y_pred.unaryExpr(sign) - y_true;
    return (diff.array() != 0).count() / N;
}