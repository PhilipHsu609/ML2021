#include "adaboost.h"

DecisionStump::DecisionStump(const Dataset &d, const std::vector<double> &u) {
    int N = d.x.size();
    double min_error = 100;  // global minimum error

    // iterate every features
    for (int f = 0; f < d.x[0].size(); ++f) {
        // step 1
        std::vector<int> idx(N, 0);  // column index
        std::vector<double> features(N, 0);

        double left_p_u = 0, left_n_u = 0;
        double right_p_u = 0, right_n_u = 0;
        for (int n = 0; n < N; ++n) {
            features[n] = d.x[n][f];

            // when theta at minus infinity, every positive/negative samples are error
            if (d.y[n] == 1)
                right_p_u += u[n];
            else
                right_n_u += u[n];
        }
        std::iota(begin(idx), end(idx), 0);
        std::sort(begin(idx), end(idx), [&features](int a, int b) -> bool { return features[a] <= features[b]; });

        // step 2
        double cur_theta = -std::numeric_limits<double>::min();
        double cur_error = 0;
        int cur_s = 0;

        // decide the direction
        if (right_p_u <= right_n_u) {
            cur_error = right_p_u;
            cur_s = -1;
        } else {
            cur_error = right_n_u;
            cur_s = 1;
        }

        if (cur_error <= min_error) {
            min_error = cur_error;
            s = cur_s;
            i = f;
            theta = cur_theta;
        }

        for (int n = 1; n < N; ++n) {
            cur_theta = (features[idx[n]] + features[idx[n - 1]]) / 2.0;

            if (d.y[idx[n - 1]] == 1) {
                left_p_u += u[idx[n - 1]];
                right_p_u -= u[idx[n - 1]];
            } else {
                left_n_u += u[idx[n - 1]];
                right_n_u -= u[idx[n - 1]];
            }

            // decide the direction
            if (left_p_u + right_n_u <= left_n_u + right_p_u) {
                cur_error = left_p_u + right_n_u;
                cur_s = 1;
            } else {
                cur_error = left_n_u + right_p_u;
                cur_s = -1;
            }

            if (cur_error <= min_error) {
                min_error = cur_error;
                s = cur_s;
                i = f;
                theta = cur_theta;
            }
        }
    }
}

int DecisionStump::operator()(const std::vector<double> &x) const {
    return s * sign(x[i] - theta);
}

AdaBoost::AdaBoost(const Dataset &d, int iter) : T{iter} {
    int N = d.x.size();

    g.clear();
    u.resize(N, 1 / static_cast<double>(N));
    alpha.resize(T, 0);

    for (int t = 0; t < T; ++t) {
        std::vector<bool> correct(N, false);

        g.push_back(DecisionStump(d, u));

        double total_u = 0, incorrect_u = 0;
        for (int n = 0; n < N; ++n) {
            if (d.y[n] == g[t](d.x[n])) {
                correct[n] = true;
            } else {
                incorrect_u += u[n];
            }
            total_u += u[n];
        }
        double eps{incorrect_u / total_u};
        double scale{std::sqrt((1 - eps) / eps)};
        for (int n = 0; n < N; ++n) {
            if (correct[n]) {
                u[n] /= scale;
            } else {
                u[n] *= scale;
            }
        }

        alpha[t] = std::log(scale);
    }
}

int AdaBoost::operator()(const std::vector<double> &x) const {
    double score = 0;
    for (int i = 0; i < g.size(); ++i) {
        score += alpha[i] * g[i](x);
    }
    return sign(score);
}