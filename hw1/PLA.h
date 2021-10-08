#ifndef PLA_H
#define PLA_H

#include <ctime>
#include <random>
#include <string>
#include <vector>

class Perceptron {
   public:
    Perceptron(std::vector<std::vector<double>> x, std::vector<int> y);
    std::vector<double> getWeights();
    void initWeights();
    void setSeed(int seed = 0);
    void fit();
    int randomPick();
    double getWeightsNormSquare();
    double accuracyRate();

   private:
    int sign(double val);
    double f(int id);
    std::mt19937 mersenne;  // mersenne twister engine
    std::vector<std::vector<double>> x;
    std::vector<int> y;
    std::vector<double> w;  // weights
    int N;                  // # of examples
    int n;                  // # of features
};

#endif