#ifndef SVM_UTILS_H
#define SVM_UTILS_H

#include <algorithm>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "svm.h"

svm_problem read_problem(const std::string &filename);

void random_sample_problem(const svm_problem &origin, svm_problem &train, svm_problem &val, int seed);

void destroy_problem(svm_problem &prob, bool release_x_space = true);

svm_problem to_binary_problem(const svm_problem &prob, int target);

svm_model *train(const svm_problem &prob, svm_parameter &param);

std::vector<int> predict(const svm_model *model, const svm_problem &prob);

double eval_error(const std::vector<int> &pred, const svm_problem &prob, int target);

double get_weight_norm(const svm_model *model);

#endif