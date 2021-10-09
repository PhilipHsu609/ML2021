#include <cmath>
#include <fstream>
#include <functional>
#include <iostream>
#include <sstream>
#include <string>

#include "PLA.h"

const std::string trainingData{"./hw1_train.dat"};

std::vector<double> split(const std::string &str) {
    std::stringstream ss{str};
    std::vector<double> v;
    double val;
    while (ss >> val) {
        v.push_back(val);
    }
    return v;
}

void readDat(const std::string &filename, std::vector<std::vector<double>> &x, std::vector<int> &y) {
    std::ifstream inFile{filename, std::ios::in};

    if (!inFile.is_open()) {
        std::cerr << "Open " << filename << " failed..." << std::endl;
        exit(-1);
    }

    while (!inFile.eof()) {
        std::string example;
        std::getline(inFile, example);

        auto v{split(example)};
        if (v.size() == 0) continue;  // encounter an empty line

        std::vector<double> tmp(v.size(), 1.0);  // x_0 = 1
        for (int i = 1; i < v.size(); i++) {
            tmp[i] = v[i - 1];
        }

        x.push_back(tmp);
        y.push_back(static_cast<int>(v.back()));
    }
    inFile.close();
}

void Q13(std::vector<std::vector<double>> &x) {
    std::cout << " -- Question 13 -- " << std::endl;
}
void Q14(std::vector<std::vector<double>> &x) {
    std::cout << " -- Question 14 -- " << std::endl;
    for (auto &example : x) {
        for (auto &val : example) {
            val *= 2;
        }
    }
}
void Q15(std::vector<std::vector<double>> &x) {
    std::cout << " -- Question 15 -- " << std::endl;
    for (auto &example : x) {
        double norm2 = 0.0;
        for (auto &val : example) {
            norm2 += val * val;
        }
        for (auto &val : example) {
            val /= sqrt(norm2);
        }
    }
}
void Q16(std::vector<std::vector<double>> &x) {
    std::cout << " -- Question 16 -- " << std::endl;
    for (auto &example : x) {
        example.front() = 0;
    }
}

void report(std::function<void(std::vector<std::vector<double>> &)> f) {
    std::vector<std::vector<double>> x;
    std::vector<int> y;

    readDat(trainingData, x, y);

    // Modifying training data to meet the question's requirements.
    f(x);

    std::cout << "x size: (" << x.size() << ", " << x[0].size() << ")\n";
    std::cout << "y size: (" << y.size() << ")\n\n";

    Perceptron p{x, y};

    double avg = 0;
    for (int i = 0; i < 1000; i++) {
        p.initWeights();
        p.setSeed();
        p.fit();
        avg += p.getWeightsNormSquare();
    }

    std::cout << "End of 1000 experiments.\n";
    std::cout << "Average of 1000 weights norm square: " << avg / 1000 << std::endl;
}

void usage() {
    std::cout << "Usage:" << std::endl;
    std::cout << "    hw1.out [13|14|15|16]" << std::endl;
    std::cout << std::endl;
}

std::vector<std::function<void(std::vector<std::vector<double>> &)>> Q = {Q13, Q14, Q15, Q16};

int main(int argc, char **argv) {
    std::cout << "Experimenting Perceptron Learning Algorithm...\n\n";

    if (argc == 2) {
        report(Q[atoi(argv[1]) - 13]);
    } else {
        usage();
    }

    return 0;
}