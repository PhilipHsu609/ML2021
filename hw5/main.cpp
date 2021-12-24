#include <chrono>
#include <functional>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "svm.h"
#include "svm_utils.h"

const std::string trainingSet{"./data/satimage.scale"};
const std::string testingSet{"./data/satimage.scale.t"};

svm_parameter param;

void Q11(svm_problem &train_prob, svm_problem &test_prob) {
    std::cout << " -- Question 11 -- " << std::endl;

    param.kernel_type = LINEAR;
    param.C = 10;

    auto five_vs_not_five{to_binary_problem(train_prob, 5)};
    auto model{train(five_vs_not_five, param)};

    std::cout << "|w| = " << get_weight_norm(model) << std::endl;

    svm_free_and_destroy_model(&model);
    destroy_problem(five_vs_not_five, false);
}

void Q12(svm_problem &train_prob, svm_problem &test_prob) {
    std::cout << " -- Question 12 -- " << std::endl;

    param.kernel_type = POLY;
    param.C = 10;
    param.coef0 = 1;
    param.gamma = 1;
    param.degree = 3;

    std::vector<svm_problem> probs{
        to_binary_problem(train_prob, 2),
        to_binary_problem(train_prob, 3),
        to_binary_problem(train_prob, 4),
        to_binary_problem(train_prob, 5),
        to_binary_problem(train_prob, 6)};

    double cur_ein = 0;
    int max_ein_prob = -1;
    for (int i = 0; i < probs.size(); ++i) {
        auto model{train(probs[i], param)};
        auto pred{predict(model, probs[i])};
        auto ein{eval_error(pred, probs[i], i + 2)};

        if (ein > cur_ein) {
            cur_ein = ein;
            max_ein_prob = i + 2;
        }

        svm_free_and_destroy_model(&model);
        destroy_problem(probs[i], false);
    }

    std::cout << max_ein_prob << " versus not " << max_ein_prob << " has the largest E_in\n";
    std::cout << "E_in = " << cur_ein << std::endl;
}

void Q13(svm_problem &train_prob, svm_problem &test_prob) {
    std::cout << " -- Question 13 -- " << std::endl;

    param.kernel_type = POLY;
    param.C = 10;
    param.coef0 = 1;
    param.gamma = 1;
    param.degree = 3;

    std::vector<svm_problem> probs{
        to_binary_problem(train_prob, 2),
        to_binary_problem(train_prob, 3),
        to_binary_problem(train_prob, 4),
        to_binary_problem(train_prob, 5),
        to_binary_problem(train_prob, 6)};

    int max_sv = 0;
    for (int i = 0; i < probs.size(); ++i) {
        auto model{train(probs[i], param)};
        auto pred{predict(model, probs[i])};
        auto ein{eval_error(pred, probs[i], i + 2)};

        max_sv = std::max(max_sv, svm_get_nr_sv(model));

        svm_free_and_destroy_model(&model);
        destroy_problem(probs[i], false);
    }

    std::cout << "Maximum number of support vectors: " << max_sv << std::endl;
}

void Q14(svm_problem &train_prob, svm_problem &test_prob) {
    std::cout << " -- Question 14 -- " << std::endl;

    param.kernel_type = RBF;
    param.gamma = 10;
    std::vector<double> Cs{0.01, 0.1, 1, 10, 100};

    auto one_vs_not_one{to_binary_problem(train_prob, 1)};

    double cur_eout = 100;
    int cur_C = -1;
    for (auto C : Cs) {
        param.C = C;

        auto model{train(one_vs_not_one, param)};
        auto pred{predict(model, test_prob)};
        auto eout{eval_error(pred, test_prob, 1)};

        if (eout < cur_eout) {
            cur_eout = eout;
            cur_C = C;
        }

        svm_free_and_destroy_model(&model);
    }
    destroy_problem(one_vs_not_one, false);

    std::cout << "C = " << cur_C << " has lowest E_out = " << cur_eout << std::endl;
}

void Q15(svm_problem &train_prob, svm_problem &test_prob) {
    std::cout << " -- Question 15 -- " << std::endl;

    param.kernel_type = RBF;
    param.C = 0.1;
    std::vector<double> gamma{0.1, 1, 10, 100, 1000};

    auto one_vs_not_one{to_binary_problem(train_prob, 1)};

    double cur_eout = 100;
    double cur_g = -1;
    for (auto g : gamma) {
        param.gamma = g;

        auto model{train(one_vs_not_one, param)};
        auto pred{predict(model, test_prob)};
        auto eout{eval_error(pred, test_prob, 1)};

        if (eout < cur_eout) {
            cur_eout = eout;
            cur_g = g;
        }

        svm_free_and_destroy_model(&model);
    }
    destroy_problem(one_vs_not_one, false);

    std::cout << "gamma = " << cur_g << " has lowest E_out = " << cur_eout << std::endl;
}

void Q16(svm_problem &train_prob, svm_problem &test_prob) {
    std::cout << " -- Question 16 -- " << std::endl;

    param.kernel_type = RBF;
    param.C = 0.1;
    std::vector<double> gamma{0.1, 1, 10, 100, 1000};
    std::vector<int> freq(5, 0);

    auto one_vs_not_one{to_binary_problem(train_prob, 1)};

    for (int i = 0; i < 1000; ++i) {
        std::cout << i + 1 << "th experiment...\n";

        int seed = static_cast<int>(
            std::chrono::high_resolution_clock::now().time_since_epoch().count());

        svm_problem train_subprob, val_prob;
        random_sample_problem(one_vs_not_one, train_subprob, val_prob, seed);

        double cur_eval = 100;
        int cur_g = -1;
        for (int j = 0; j < gamma.size(); ++j) {
            param.gamma = gamma[j];

            auto model{train(train_subprob, param)};
            auto pred{predict(model, val_prob)};
            auto eval{eval_error(pred, val_prob, 1)};

            if (eval < cur_eval) {
                cur_eval = eval;
                cur_g = j;
            }

            svm_free_and_destroy_model(&model);
        }

        ++freq[cur_g];
        destroy_problem(train_subprob, false);
        destroy_problem(val_prob, false);
    }

    destroy_problem(one_vs_not_one, false);

    std::cout << "(gamma, count): ";
    for (int i = 0; i < freq.size(); ++i)
        std::cout << "(" << gamma[i] << ", " << freq[i] << ") ";
    std::cout << std::endl;

    int idx = std::distance(begin(freq), std::max_element(begin(freq), end(freq)));
    std::cout << "gamma = " << gamma[idx] << " is selected the most time\n";
}

std::vector<std::function<void(svm_problem &, svm_problem &)>> Q = {Q11, Q12, Q13, Q14, Q15, Q16};

void usage() {
    std::cout << "Usage:" << std::endl;
    std::cout << "    ./hw5.out [11|12|13|14|15|16]" << std::endl;
    std::cout << std::endl;
}

int main(int argc, char **argv) {
    std::cout << "Experimenting Soft-Margin SVM...\n\n";

    auto train_prob{read_problem(trainingSet)};
    auto test_prob{read_problem(testingSet)};

    // default parameters
    param.svm_type = C_SVC;
    param.kernel_type = LINEAR;
    param.degree = 3;
    param.gamma = 1 / 36.0;  // 1/num_features
    param.coef0 = 0;
    param.nu = 0.5;
    param.cache_size = 1024;
    param.C = 1;
    param.eps = 1e-3;
    param.p = 0.1;
    param.shrinking = 1;
    param.probability = 0;
    param.nr_weight = 0;
    param.weight_label = NULL;
    param.weight = NULL;

    // quiet
    svm_set_print_string_function([](const char *) {});

    if (argc == 2) {
        Q[atoi(argv[1]) - 11](train_prob, test_prob);
    } else {
        usage();
    }

    svm_destroy_param(&param);
    destroy_problem(train_prob);
    destroy_problem(test_prob);

    return 0;
}