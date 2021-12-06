#include <Eigen/Dense>
#include <fstream>
#include <iostream>
#include <vector>

#include "Features.h"

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

int main(int argc, char **argv) {
    // Feature transform
    // argv[1]: input file name
    // argv[2]: output file name

    if (argc < 3) {
        std::cout << "Need two arguments...\n";
        exit(-1);
    }

    std::string inputFileName{argv[1]};
    std::string outputFileName{argv[2]};

    auto data{readDat(inputFileName)};

    std::cout << inputFileName << ": (" << data.rows() << ", " << data.cols() << ")\n";

    auto X{data(Eigen::all, Eigen::seq(0, Eigen::last - 1))};
    auto y{data.col(data.cols() - 1)};
    auto output{polynomialFeatures(X, 3)};

    std::ofstream outFile{outputFileName, std::ios::out};

    for (int i = 0; i < output.rows(); i++) {
        outFile << y[i];
        for (int j = 0; j < output.cols(); j++) {
            outFile << ' ' << j + 1 << ':' << output(i, j);
        }
        outFile << '\n';
    }

    outFile.close();
    std::cout << "Transformed size: (" << output.rows() << ", " << output.cols() << ").\n";

    return 0;
}