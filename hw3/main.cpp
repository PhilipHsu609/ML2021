#include <Eigen/Dense>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "Features.h"
#include "Regression.h"

const std::string trainingData{"./hw3_train.dat"};
const std::string testingData{"./hw3_test.dat"};

std::vector<double> split(const std::string &str) {
    std::stringstream ss{str};
    std::vector<double> v;
    double val;
    while (ss >> val) {
        v.push_back(val);
    }
    return v;
}

Eigen::MatrixXd readDat(const std::string &filename) {
    std::ifstream inFile{filename, std::ios::in};

    if (!inFile.is_open()) {
        std::cerr << "Open " << filename << " failed..." << std::endl;
        exit(-1);
    }

    std::vector<std::vector<double>> buf;
    while (!inFile.eof()) {
        std::string sample;
        std::getline(inFile, sample);

        auto v{split(sample)};
        if (v.size() == 0) continue;  // encounter an empty line

        std::vector<double> tmp(v.size(), 0);
        for (int i = 0; i < v.size(); i++) {
            tmp[i] = v[i];
        }

        buf.push_back(tmp);
    }
    inFile.close();

    Eigen::MatrixXd data(buf.size(), buf[0].size());
    for (int i = 0; i < buf.size(); i++) {
        data.row(i) = Eigen::Map<Eigen::VectorXd>(buf[i].data(), buf[i].size());
    }
    return data;
}

void Q12(Eigen::MatrixXd &X_train, Eigen::VectorXd &y_train, Eigen::MatrixXd &X_test, Eigen::VectorXd &y_test) {
    std::cout << " -- Question 12 -- " << std::endl;
    LinearRegression l;

    auto Z_train{homogeneousPolynomialFeatures(X_train, 2)};
    auto Z_test{homogeneousPolynomialFeatures(X_test, 2)};

    l.fit(Z_train, y_train);
    double ein{ErrorMeasure::zeroOneError(l.predict(Z_train), y_train)};
    double eout{ErrorMeasure::zeroOneError(l.predict(Z_test), y_test)};

    std::cout << "Ans: " << std::abs(ein - eout) << std::endl;
}

void Q13(Eigen::MatrixXd &X_train, Eigen::VectorXd &y_train, Eigen::MatrixXd &X_test, Eigen::VectorXd &y_test) {
    std::cout << " -- Question 13 -- " << std::endl;
    LinearRegression l;

    auto Z_train{homogeneousPolynomialFeatures(X_train, 8)};
    auto Z_test{homogeneousPolynomialFeatures(X_test, 8)};

    l.fit(Z_train, y_train);
    double ein{ErrorMeasure::zeroOneError(l.predict(Z_train), y_train)};
    double eout{ErrorMeasure::zeroOneError(l.predict(Z_test), y_test)};

    std::cout << "Ans: " << std::abs(ein - eout) << std::endl;
}

void Q14(Eigen::MatrixXd &X_train, Eigen::VectorXd &y_train, Eigen::MatrixXd &X_test, Eigen::VectorXd &y_test) {
    std::cout << " -- Question 14 -- " << std::endl;
    LinearRegression l;

    auto Z_train{polynomialFeatures(X_train, 2)};
    auto Z_test{polynomialFeatures(X_test, 2)};

    l.fit(Z_train, y_train);
    double ein{ErrorMeasure::zeroOneError(l.predict(Z_train), y_train)};
    double eout{ErrorMeasure::zeroOneError(l.predict(Z_test), y_test)};

    std::cout << "Ans: " << std::abs(ein - eout) << std::endl;
}

void Q15(Eigen::MatrixXd &X_train, Eigen::VectorXd &y_train, Eigen::MatrixXd &X_test, Eigen::VectorXd &y_test) {
    std::cout << " -- Question 15 -- " << std::endl;
    LinearRegression l;

    double cur_min = 1e5;
    int ans = -1;
    for (int i = 1; i <= 10; i++) {
        auto Z_train{lessFeatures(X_train, i)};
        auto Z_test{lessFeatures(X_test, i)};

        l.fit(Z_train, y_train);
        double ein{ErrorMeasure::zeroOneError(l.predict(Z_train), y_train)};
        double eout{ErrorMeasure::zeroOneError(l.predict(Z_test), y_test)};

        double diff{std::abs(ein - eout)};
        if (cur_min > diff) {
            ans = i;
            cur_min = diff;
        }
    }
    std::cout << "Ans: " << ans << std::endl;
}

void Q16(Eigen::MatrixXd &X_train, Eigen::VectorXd &y_train, Eigen::MatrixXd &X_test, Eigen::VectorXd &y_test) {
    std::cout << " -- Question 16 -- " << std::endl;
    LinearRegression l;

    double avg = 0;
    for (int i = 0; i < 200; i++) {
        auto Z_train{randomFeatures(X_train, 5)};
        auto Z_test{randomFeatures(X_test, 5)};

        l.fit(Z_train, y_train);
        double ein{ErrorMeasure::zeroOneError(l.predict(Z_train), y_train)};
        double eout{ErrorMeasure::zeroOneError(l.predict(Z_test), y_test)};
        avg += std::abs(ein - eout);
    }

    avg /= 200;
    std::cout << "Ans: " << avg << std::endl;
}

void usage() {
    std::cout << "Usage:" << std::endl;
    std::cout << "    ./hw2.out [12|13|14|15|16]" << std::endl;
    std::cout << std::endl;
}

std::vector<std::function<void(Eigen::MatrixXd &, Eigen::VectorXd &, Eigen::MatrixXd &, Eigen::VectorXd &)>> Q = {Q12, Q13, Q14, Q15, Q16};

int main(int argc, char **argv) {
    auto train{readDat(trainingData)};
    auto test{readDat(testingData)};

    Eigen::VectorXd y_train{train.col(train.cols() - 1)}, y_test{test.col(test.cols() - 1)};
    Eigen::MatrixXd X_train{train(Eigen::all, Eigen::seq(0, Eigen::last - 1))};
    Eigen::MatrixXd X_test{test(Eigen::all, Eigen::seq(0, Eigen::last - 1))};

    std::cout << "Experimenting with Linear and Nonlinear Models...\n\n";

    if (argc == 2) {
        Q[atoi(argv[1]) - 12](X_train, y_train, X_test, y_test);
    } else {
        usage();
    }

    return 0;
}