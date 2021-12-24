#include "svm_utils.h"

svm_problem read_problem(const std::string &filename) {
    std::ifstream f{filename, std::ios::in};
    std::string line;
    int elements = 0;

    if (!f.is_open()) {
        std::cerr << "Open " << filename << " failed..." << std::endl;
        exit(1);
    }

    svm_problem prob;
    svm_node *x_space;

    prob.l = 0;
    while (!f.eof()) {
        std::getline(f, line);

        if (line.empty())
            break;

        std::string tmp;
        std::stringstream ss{line};
        ss >> tmp;  // label

        // features
        while (ss >> tmp) {
            ++elements;
        }
        ++elements;
        ++prob.l;
    }
    f.clear();
    f.seekg(0, f.beg);

    prob.y = new double[prob.l];
    prob.x = new svm_node *[prob.l];
    x_space = new svm_node[elements];

    int j = 0;
    for (int i = 0; i < prob.l; ++i) {
        std::getline(f, line);
        std::string tmp;
        std::stringstream ss{line};

        if (line.empty())
            break;

        ss >> tmp;  // label
        prob.x[i] = &x_space[j];
        prob.y[i] = std::stod(tmp);

        // features
        while (ss >> tmp) {
            int sep = tmp.find(':');
            x_space[j].index = std::stoi(tmp.substr(0, sep));
            x_space[j].value = std::stod(tmp.substr(sep + 1));
            ++j;
        }
        x_space[j++].index = -1;
    }

    f.close();

    return prob;
}

void random_sample_problem(const svm_problem &origin, svm_problem &train, svm_problem &val, int seed) {
    // Randomly sample 200 samples from 'origin' problem to 'val' problem, the rest goes to 'train' problem
    // x in 'train' and 'val' are array of pointers pointing to the paritial of x in 'origin'
    constexpr int val_size = 200;

    train.l = origin.l - val_size;
    val.l = val_size;

    train.y = new double[train.l];
    val.y = new double[val.l];

    train.x = new svm_node *[train.l];
    val.x = new svm_node *[val.l];

    // randomly pick 200 samples for validation, and sort the index in ascending order
    std::mt19937 mersenne{seed};
    std::vector<int> idx(origin.l);
    std::iota(begin(idx), end(idx), 0);
    std::shuffle(begin(idx), end(idx), mersenne);
    std::sort(begin(idx), begin(idx) + val_size);

    int train_idx = 0, val_idx = 0;
    for (int i = 0; i < origin.l; ++i) {
        if (val_idx < val_size && i == idx[val_idx]) {
            // for validation
            val.y[val_idx] = origin.y[i];
            val.x[val_idx] = origin.x[i];
            ++val_idx;
        } else {
            // for training
            train.y[train_idx] = origin.y[i];
            train.x[train_idx] = origin.x[i];
            ++train_idx;
        }
    }
}

void destroy_problem(svm_problem &prob, bool release_x_space) {
    delete[] prob.y;
    if (release_x_space)
        delete[] prob.x[0];  // release x_space allocated in raed_problem
    delete[] prob.x;
}

svm_problem to_binary_problem(const svm_problem &prob, int target) {
    // 'target' versus 'non target'
    svm_problem binary_prob;
    binary_prob.l = prob.l;
    binary_prob.y = new double[prob.l];
    binary_prob.x = new svm_node *[prob.l];
    for (int i = 0; i < prob.l; ++i) {
        binary_prob.y[i] = prob.y[i] == target ? target : -target;
        binary_prob.x[i] = prob.x[i];
    }
    return binary_prob;
}

svm_model *train(const svm_problem &prob, svm_parameter &param) {
    const char *error_msg = svm_check_parameter(&prob, &param);

    if (error_msg) {
        std::cerr << "ERROR: " << error_msg << std::endl;
        exit(1);
    }

    auto model{svm_train(&prob, &param)};
    return model;
}

std::vector<int> predict(const svm_model *model, const svm_problem &prob) {
    std::vector<int> pred;

    for (int i = 0; i < prob.l; ++i) {
        pred.push_back(static_cast<int>(svm_predict(model, prob.x[i])));
    }

    return pred;
}

double eval_error(const std::vector<int> &pred, const svm_problem &prob, int target) {
    // prob: multiclass{1,2,...,6}, pred: binary class{target, -target}
    int correct = 0, n = pred.size();
    for (int i = 0; i < n; ++i) {
        int y = static_cast<int>(prob.y[i]);
        if (y == pred[i])
            ++correct;
    }
    return (n - correct) / static_cast<double>(n);
}

double get_weight_norm(const svm_model *model) {
    /*
        Get the the norm of the weight vector, only for linear kernel
        w = \sum \alpha_n * y_n * x^T x
        w = SVs' * coef

        model.l = # of SVs
        model.SV = SVs
        model.sv_coef = y *\alpha
    */
    if (model->param.kernel_type != LINEAR) {
        std::cerr << "SVM model isn't using linear kernel...\n";
        exit(-1);
    }

    int nr_sv, nr_class;
    nr_class = svm_get_nr_class(model);
    nr_sv = svm_get_nr_sv(model);

    double **sv_coef{model->sv_coef};
    svm_node **sv{model->SV};

    int max_feature = 0;
    for (int i = 0; i < nr_sv; ++i) {
        int j = 0;
        while (sv[i][j].index != -1) {
            max_feature = std::max(max_feature, sv[i][j].index);
            ++j;
        }
    }

    std::vector<double> w(max_feature, 0);
    for (int i = 0; i < nr_sv; ++i) {
        int j = 0;
        while (sv[i][j].index != -1) {
            w[sv[i][j].index - 1] += sv_coef[0][i] * sv[i][j].value;
            ++j;
        }
    }
    double norm = std::inner_product(begin(w), end(w), begin(w), 0.0);

    return std::sqrt(norm);
}