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