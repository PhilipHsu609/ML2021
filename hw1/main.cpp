#include <fstream>
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

void readDat(const std::string &filename, std::vector<std::vector<double>> &x, std::vector<int> &y, double x0 = 1.0) {
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

        std::vector<double> tmp(v.size(), x0);
        for (int i = 1; i < v.size(); i++) {
            tmp[i] = v[i - 1];
        }

        x.push_back(tmp);
        y.push_back(static_cast<int>(v.back()));
    }
}

int main() {
    std::cout << "Experimenting Perceptron Learning Algorithm...\n\n";
    std::vector<std::vector<double>> x;
    std::vector<int> y;

    readDat(trainingData, x, y);
    std::cout << "Reading training data...\n";

    std::cout << "x size: (" << x.size() << ", " << x[0].size() << ")\n";
    std::cout << "y size: (" << y.size() << ")\n\n";

    std::cout << "x_0 = [";
    for (auto &i : x[0]) {
        std::cout << i << " ";
    }
    std::cout << "], y_0 = " << y.front() << std::endl;
    std::cout << std::endl;

    Perceptron p{x, y};
    p.fit();

    std::cout << "End of training.\n";
    std::cout << "Accuracy rate for training set: " << p.accuracyRate() << std::endl;
    std::cout << "Norm square of weights: " << p.getWeightsNormSquare() << std::endl;
    return 0;
}