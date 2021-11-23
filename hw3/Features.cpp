#include "Features.h"

int C(int n, int r) {
    int ans = 1;
    for (int i = 1; i <= r; i++) {
        ans *= n - r + i;
        ans /= i;
    }
    return ans;
}

Eigen::MatrixXd polynomialFeatures(const Eigen::MatrixXd &X, int degree) {
    /*
        Ref: sklearn.preprocessing.PolynomialFeatures
        https://github.com/scikit-learn/scikit-learn/blob/0d378913b/sklearn/preprocessing/_polynomial.py#L452
    */
    Eigen::MatrixXd features(X.rows(), C(degree + X.cols(), degree));

    features.col(0) = Eigen::VectorXd::Ones(X.rows());
    features(Eigen::all, Eigen::seq(1, X.cols())) = X;

    int current_col = 1;
    int n_features = X.cols();

    std::vector<int> index(n_features);
    std::iota(begin(index), end(index), current_col);

    current_col += n_features;
    index.push_back(current_col);

    for (int d = 2; d < degree + 1; d++) {
        std::vector<int> new_index;
        int end = index.back();

        for (int feature_idx = 0; feature_idx < n_features; feature_idx++) {
            int start = index[feature_idx];
            new_index.push_back(current_col);

            int next_col = current_col + end - start;
            if (next_col <= current_col) break;

            features(Eigen::all, Eigen::seq(current_col, next_col - 1)) =
                features(Eigen::all, Eigen::seq(start, end - 1)).array().colwise() * X.col(feature_idx).array();

            current_col = next_col;
        }
        new_index.push_back(current_col);
        index = new_index;
    }

    return features;
}

Eigen::MatrixXd homogeneousPolynomialFeatures(const Eigen::MatrixXd &X, int degree) {
    Eigen::MatrixXd features(X.rows(), X.cols() * degree + 1);

    features.col(0) = Eigen::VectorXd::Ones(X.rows());
    features(Eigen::all, Eigen::seq(1, X.cols())) = X;

    for (int d = 1; d < degree; d++) {
        int current_col = d * X.cols() + 1, next_col = (d + 1) * X.cols();
        features(Eigen::all, Eigen::seq(current_col, next_col)) =
            features(Eigen::all, Eigen::seq(1, X.cols())).array().pow(d + 1);
    }

    return features;
}

Eigen::MatrixXd lessFeatures(const Eigen::MatrixXd &X, int n) {
    Eigen::MatrixXd features(X.rows(), n + 1);
    features.col(0) = Eigen::VectorXd::Ones(X.rows());
    features(Eigen::all, Eigen::seq(1, n)) = X(Eigen::all, Eigen::seq(0, n - 1));
    return features;
}

Eigen::MatrixXd randomFeatures(const Eigen::MatrixXd &X, int n, int seed) {
    std::mt19937 mersene(seed);

    Eigen::MatrixXd features(X.rows(), n + 1);
    features.col(0) = Eigen::VectorXd::Ones(X.rows());

    std::vector<int> choice(X.cols());
    std::iota(begin(choice), end(choice), 0);
    std::shuffle(begin(choice), end(choice), mersene);

    for (int i = 1; i <= n; i++) {
        features.col(i) = X.col(choice[i - 1]);
    }

    return features;
}