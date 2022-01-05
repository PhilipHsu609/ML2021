#ifndef ADABOOST_H
#define ADABOOST_H

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <vector>

struct Dataset {
    std::vector<std::vector<double>> x;
    std::vector<int> y;
};

inline int sign(double x) {
    return x >= 0 ? 1 : -1;
}

class DecisionStump {
   public:
    DecisionStump(const Dataset &d, const std::vector<double> &u);
    int operator()(const std::vector<double> &x) const;

   private:
    int s{}, i{};
    double theta{};
};

class AdaBoost {
   public:
    AdaBoost(const Dataset &d, int iter);
    int operator()(const std::vector<double> &x) const;
    std::vector<DecisionStump> g;

   private:
    int T{};
    std::vector<double> u, alpha;
};

#endif