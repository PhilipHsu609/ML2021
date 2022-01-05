#include <fstream>
#include <functional>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "adaboost.h"

const std::string trainingData{"./hw6_train.dat"};
const std::string testingData{"./hw6_test.dat"};

std::vector<double> split(const std::string &str) {
    std::stringstream ss{str};
    std::vector<double> v;
    double val;
    while (ss >> val) {
        v.push_back(val);
    }
    return v;
}

Dataset readDat(const std::string &filename) {
    std::ifstream inFile{filename, std::ios::in};

    if (!inFile.is_open()) {
        std::cerr << "Open " << filename << " failed..." << std::endl;
        exit(-1);
    }

    Dataset d;
    while (!inFile.eof()) {
        std::string example;
        std::getline(inFile, example);

        auto v{split(example)};
        if (v.size() == 0) continue;  // encounter an empty line

        int label = v.back();
        v.pop_back();

        d.x.push_back(v);
        d.y.push_back(label);
    }
    inFile.close();

    return d;
}

void Q11(const AdaBoost &ada, const Dataset &d) {
    std::cout << " -- Question 11 -- " << std::endl;

    int N = d.x.size(), ein = 0;
    for (int n = 0; n < N; ++n) {
        if (ada.g[0](d.x[n]) != d.y[n])
            ++ein;
    }

    std::cout << ein / static_cast<double>(N) << std::endl;
}

void Q12(const AdaBoost &ada, const Dataset &d) {
    std::cout << " -- Question 12 -- " << std::endl;

    int N = d.x.size(), max_ein = 0;
    for (int t = 0; t < ada.g.size(); ++t) {
        int ein = 0;
        for (int n = 0; n < N; ++n) {
            if (ada.g[t](d.x[n]) != d.y[n])
                ++ein;
        }
        max_ein = std::max(max_ein, ein);
    }

    std::cout << max_ein / static_cast<double>(N) << std::endl;
}

void Q13(const AdaBoost &ada, const Dataset &d) {
    std::cout << " -- Question 13 -- " << std::endl;

    std::vector<int> choices{60, 160, 260, 360, 460};

    int N = d.x.size();
    for (int t : choices) {
        int min_ein = std::numeric_limits<int>::max();

        for (int tau = 1; tau <= t; tau++) {
            AdaBoost G(d, tau);
            int ein = 0;
            for (int n = 0; n < N; ++n) {
                if (G(d.x[n]) != d.y[n])
                    ++ein;
            }
            min_ein = std::min(min_ein, ein);

            if (min_ein / static_cast<double>(N) <= 0.05) {
                std::cout << t << std::endl;
                return;
            }
        }
    }
}

void Q14(const AdaBoost &ada, const Dataset &d) {
    std::cout << " -- Question 14 -- " << std::endl;

    int N = d.x.size(), ein = 0;
    for (int n = 0; n < N; ++n) {
        if (ada.g[0](d.x[n]) != d.y[n])
            ++ein;
    }

    std::cout << ein / static_cast<double>(N) << std::endl;
}

void Q15(const AdaBoost &ada, const Dataset &d) {
    std::cout << " -- Question 15 -- " << std::endl;

    auto G{[&ada](const std::vector<double> &x) -> int {
        double score = 0;

        for (int t = 0; t < ada.g.size(); ++t) {
            score += ada.g[t](x);
        }

        return sign(score);
    }};

    int eout = 0, N = d.x.size();
    for (int n = 0; n < N; ++n) {
        if (G(d.x[n]) != d.y[n])
            ++eout;
    }

    std::cout << eout / static_cast<double>(N) << std::endl;
}

void Q16(const AdaBoost &ada, const Dataset &d) {
    std::cout << " -- Question 16 -- " << std::endl;

    int eout = 0, N = d.x.size();
    for (int n = 0; n < N; ++n) {
        if (ada(d.x[n]) != d.y[n])
            ++eout;
    }

    std::cout << eout / static_cast<double>(N) << std::endl;
}

void usage() {
    std::cout << "Usage:" << std::endl;
    std::cout << "    ./hw6.out [11|12|13|14|15|16]" << std::endl;
    std::cout << std::endl;
}

std::vector<std::function<void(const AdaBoost &, const Dataset &)>> Q{Q11, Q12, Q13, Q14, Q15, Q16};

int main(int argc, char **argv) {
    Dataset train{readDat(trainingData)}, test{readDat(testingData)};

    std::cout << "Experimenting with Adaptive Boosting...\n\n";

    AdaBoost ada(train, 500);

    if (argc == 2) {
        int idx{atoi(argv[1]) - 11};
        if (idx < 3)
            Q[idx](ada, train);
        else
            Q[idx](ada, test);
    } else {
        usage();
    }

    return 0;
}