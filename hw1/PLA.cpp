#include "PLA.h"

Perceptron::Perceptron(std::vector<std::vector<double>> x, std::vector<int> y)
    : x{x},
      y{y},
      N{static_cast<int>(x.size())},
      n{static_cast<int>(x[0].size())} {
    setSeed();
    initWeights();
}

void Perceptron::initWeights() {
    w.clear();
    w.resize(n, 0);
}

std::vector<double> Perceptron::getWeights() {
    return w;
}

int Perceptron::sign(double val) {
    return val > 0 ? 1 : -1;
}

void Perceptron::setSeed(int seed) {
    if (seed == 0) {
        seed = static_cast<int>(
            std::chrono::high_resolution_clock::now().time_since_epoch().count());
    }
    mersenne.seed(static_cast<std::mt19937::result_type>(seed));
}

int Perceptron::randomPick() {
    std::uniform_int_distribution<> example{0, N - 1};
    return example(mersenne);
}

double Perceptron::f(int id) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += x[id][i] * w[i];
    }
    return sum;
}

double Perceptron::getWeightsNormSquare() {
    double norm2 = 0.0;
    for (int i = 0; i < n; i++) {
        norm2 += w[i] * w[i];
    }
    return norm2;
}

void Perceptron::fit() {
    int correctSteps = 0;
    while (true) {
        // End of training.
        if (correctSteps == 5 * N) {
            break;
        }

        int id{randomPick()};
        // Wrong example
        if (sign(f(id)) != y[id]) {
            correctSteps = 0;

            // update weights
            for (int i = 0; i < n; i++) {
                w[i] += y[id] * x[id][i];
            }
        } else {
            correctSteps++;
        }
    }
}

double Perceptron::accuracyRate() {
    int correct = 0;
    for (int i = 0; i < N; i++) {
        if (sign(f(i)) == y[i])
            correct++;
    }
    return correct / N;
}