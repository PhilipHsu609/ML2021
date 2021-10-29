#include "Regression.h"

Dataset::Dataset() { setSeed(); }

void Dataset::setSeed(int seed) {
    if (seed == 0) {
        seed = static_cast<int>(
            std::chrono::high_resolution_clock::now().time_since_epoch().count());
    }
    mersenne.seed(static_cast<std::mt19937::result_type>(seed));
}

int Dataset::flipCoin() {
    // Return 1 or -1
    std::uniform_int_distribution<> coin(0, 1);
    return coin(mersenne) * 2 - 1;
}

Eigen::Vector3d Dataset::singleSample(int y) {
    Eigen::Vector3d sample;

    // std::normal_distribution needs mean and standard deviation
    if (y == 1) {
        std::normal_distribution<double> n1(2, std::sqrt(0.6)), n2(3, std::sqrt(0.6));
        sample << 1, n1(mersenne), n2(mersenne);
    } else {
        std::normal_distribution<double> n1(0, std::sqrt(0.4)), n2(4, std::sqrt(0.4));
        sample << 1, n1(mersenne), n2(mersenne);
    }

    // sample size: (3, 1)
    return sample;
}

void Dataset::samples(int trainSize, int testSize) {
    X_train.resize(trainSize, 3), X_test.resize(testSize, 3);
    y_train.resize(trainSize, 1), y_test.resize(testSize, 1);

    for (int i = 0; i < trainSize; i++) {
        int label = flipCoin();
        X_train.row(i) = singleSample(label);
        y_train[i] = label;
    }

    for (int i = 0; i < testSize; i++) {
        int label = flipCoin();
        X_test.row(i) = singleSample(label);
        y_test[i] = label;
    }
}

void Dataset::addOutliers(int size) {
    // Add outlier examples to the training data
    int currentSamples = X_train.rows();
    X_train.conservativeResize(currentSamples + size, 3);
    y_train.conservativeResize(currentSamples + size, 1);

    std::normal_distribution<float> n1(6, std::sqrt(0.3)), n2(0, std::sqrt(0.1));
    for (int i = 0; i < size; i++) {
        X_train.row(currentSamples + i) = Eigen::Vector3d(1, n1(mersenne), n2(mersenne));
        y_train[currentSamples + i] = 1;
    }
}

void LinearRegression::fit(Eigen::MatrixXd &X, Eigen::VectorXd &y) {
    Eigen::MatrixXd pinv = X.completeOrthogonalDecomposition().pseudoInverse();
    w = pinv * y;
}

Eigen::VectorXd LinearRegression::predict(Eigen::MatrixXd &X) {
    return X * w;
}

double ErrorMeasure::squareError(Eigen::VectorXd &y_pred, Eigen::VectorXd &y_true) {
    double N = y_pred.rows();
    return (y_pred - y_true).squaredNorm() / N;
}

double ErrorMeasure::zeroOneError(Eigen::VectorXd &y_pred, Eigen::VectorXd &y_true) {
    double N = y_pred.rows();

    std::function<double(double)> sign = [](double val) -> double {
        return val >= 0 ? 1.0 : -1.0;
    };

    Eigen::VectorXd diff = y_pred.unaryExpr(sign) - y_true;
    return (diff.array() != 0).count() / N;
}

LogisticRegression::LogisticRegression(double eta, int T) : eta{eta}, T{T} {}

void LogisticRegression::fit(Eigen::MatrixXd &X, Eigen::VectorXd &y) {
    w = Eigen::VectorXd::Zero(3);

    std::function<double(double)> sigmoid = [](double x) -> double {
        return 1 / (1 + std::exp(-x));
    };

    for (int i = 0; i < T; i++) {
        Eigen::VectorXd grad = Eigen::VectorXd::Zero(3);

        for (int j = 0; j < X.rows(); j++) {
            grad += (sigmoid(-y[j] * X.row(j).dot(w)) * -y[j]) * X.row(j).transpose();
        }

        grad /= X.rows();
        w -= eta * grad;
    }
}

Eigen::VectorXd LogisticRegression::predict(Eigen::MatrixXd &X) {
    return X * w;
}