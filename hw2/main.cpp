#include <Eigen/Dense>
#include <Eigen/QR>
#include <iostream>
#include <vector>

#include "Regression.h"

void Q13() {
    std::cout << " -- Question 13 -- " << std::endl;
    double avg = 0;
    Dataset D;
    LinearRegression L;
    for (int i = 0; i < 100; i++) {
        D.setSeed();
        D.samples();
        L.fit(D.X_train, D.y_train);
        Eigen::VectorXd y_in = L.predict(D.X_train);
        avg += L.squareError(y_in, D.y_train);
    }
    avg /= 100;
    std::cout << "Ans: " << avg << std::endl;
}

void Q14() {
    std::cout << " -- Question 14 -- " << std::endl;
    double avg = 0;
    Dataset D;
    LinearRegression L;
    for (int i = 0; i < 100; i++) {
        D.setSeed();
        D.samples();
        L.fit(D.X_train, D.y_train);
        Eigen::VectorXd y_in = L.predict(D.X_train);
        Eigen::VectorXd y_out = L.predict(D.X_test);
        avg += std::abs(L.zeroOneError(y_in, D.y_train) - L.zeroOneError(y_out, D.y_test));
    }
    avg /= 100;
    std::cout << "Ans: " << avg << std::endl;
}

void Q15() {
    std::cout << " -- Question 15 -- " << std::endl;
    double avgA = 0, avgB = 0;
    Dataset D;
    LinearRegression A;
    LogisticRegression B(0.1, 500);
    for (int i = 0; i < 100; i++) {
        D.setSeed();
        D.samples();
        A.fit(D.X_train, D.y_train);
        B.fit(D.X_train, D.y_train);
        Eigen::VectorXd y_A = A.predict(D.X_test);
        Eigen::VectorXd y_B = B.predict(D.X_test);
        avgA += A.zeroOneError(y_A, D.y_test);
        avgB += B.zeroOneError(y_B, D.y_test);
    }
    avgA /= 100;
    avgB /= 100;
    std::cout << "Ans: [" << avgA << ", " << avgB << "]\n";
}

void Q16() {
    std::cout << " -- Question 16 -- " << std::endl;
    double avgA = 0, avgB = 0;
    Dataset D;
    LinearRegression A;
    LogisticRegression B(0.1, 500);
    for (int i = 0; i < 100; i++) {
        D.setSeed();
        D.samples();
        D.addOutliers();
        A.fit(D.X_train, D.y_train);
        B.fit(D.X_train, D.y_train);
        Eigen::VectorXd y_A = A.predict(D.X_test);
        Eigen::VectorXd y_B = B.predict(D.X_test);
        avgA += A.zeroOneError(y_A, D.y_test);
        avgB += B.zeroOneError(y_B, D.y_test);
    }
    avgA /= 100;
    avgB /= 100;
    std::cout << "Ans: [" << avgA << ", " << avgB << "]\n";
}

void usage() {
    std::cout << "Usage:" << std::endl;
    std::cout << "    ./hw2.out [13|14|15|16]" << std::endl;
    std::cout << std::endl;
}

std::vector<std::function<void()>> Q = {Q13, Q14, Q15, Q16};

int main(int argc, char **argv) {
    std::cout << "Experimenting Linear Models...\n\n";

    if (argc == 2) {
        Q[atoi(argv[1]) - 13]();
    } else {
        usage();
    }

    return 0;
}