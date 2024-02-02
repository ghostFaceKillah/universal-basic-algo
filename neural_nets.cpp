#include <iostream>
#include <Eigen/Dense>
#include <functional>
#include <random>

void example_more_eigen_stuff() {
    // Eigen::MatrixXf m(4, 4);
    // m << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,14,15,16;

    // Eigen::MatrixXf m(4, 4);

    // Eigen::MatrixXf m = Eigen::MatrixXf::Constant(4, 4, 1.0);
    Eigen::MatrixXf m = Eigen::MatrixXf::Ones(4, 4);

    std::cout << "block in the middle" << std::endl;
    std::cout << m.block<2, 2>(1, 1) << std::endl;
}


Eigen::MatrixXd
draw_weights(
    int h,
    int w,
    std::function<double()> rand_gen
) {
    Eigen::MatrixXd matrix(h, w);

    for (int i = 0; i < h; ++i) {
        for (int j = 0; j < w; ++j) {
            matrix(i, j) = rand_gen();
        }
    }

    return matrix;
}

constexpr int kEpochs = 5000;
// constexpr int kEpochs = 5;
constexpr int kSampleSize = 100;
// constexpr int kSampleSize = 3;
const double kVar = 1.0; // Variance of data generating process
const double kLearningRate = 0.01;

const Eigen::Vector2d kLowerLeftZerosCenter(-2, -2);
const Eigen::Vector2d kUpperLeftOnesCenter(-2, 2);
const Eigen::Vector2d kLowerRightOnesCenter(2, -2);
const Eigen::Vector2d kUpperRightZerosCenter(2, 2);

Eigen::MatrixXd draw_sample_helper(
    const Eigen::Vector2d &mean,
    double std,
    int sample_size,
    std::function<double(double, double)> rand_gen
) {
    Eigen::MatrixXd sample(sample_size, 2);

    for (int i = 0; i < sample_size; ++i) {
        sample(i, 0) = rand_gen(mean(0), std);
        sample(i, 1) = rand_gen(mean(1), std);
    }

    return sample;
}

std::pair<Eigen::MatrixXd, Eigen::VectorXd>
draw_sample(
    int sample_size,
    std::function<double(double, double)> rand_gen
) {
    // prepare x
    Eigen::MatrixXd x = Eigen::MatrixXd::Zero(4 * sample_size, 2);
    x.topRows(sample_size) =
            draw_sample_helper(kLowerRightOnesCenter, sqrt(kVar), sample_size, rand_gen);
    x.middleRows(sample_size, sample_size) =
            draw_sample_helper(kUpperLeftOnesCenter, sqrt(kVar), sample_size, rand_gen);
    x.middleRows(2 * sample_size, sample_size) =
            draw_sample_helper(kLowerLeftZerosCenter, sqrt(kVar), sample_size, rand_gen);
    x.bottomRows(sample_size) =
            draw_sample_helper(kUpperRightZerosCenter, sqrt(kVar), sample_size, rand_gen);

    // prepare y
    Eigen::VectorXd y(4 * sample_size);
    y.head(2 * sample_size) = Eigen::VectorXd::Ones(2 * sample_size);
    y.tail(2 * sample_size) = Eigen::VectorXd::Zero(2 * sample_size);

    return std::make_pair(x, y);
}

Eigen::MatrixXd relu(const Eigen::MatrixXd &matrix) {
    return matrix.unaryExpr([](double x) { return std::max(0.0, x); });
}

Eigen::MatrixXd sigmoid(const Eigen::MatrixXd &matrix) {
    return matrix.unaryExpr([](double x) {
        return 1 / (1 + std::exp(-x));
    });
}


void neural_networks_example() {
    srandom((unsigned int) time(0));

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> dis(0, 1);

    auto rand_normal = [&](double mean, double std) -> double { return std * dis(gen) + mean; };
    auto unit_normal = [&]() -> double { return rand_normal(0, 1); };

    // Eigen::MatrixXd weights = Eigen::MatrixXd::Random(4, 2);
    Eigen::MatrixXd W_1 = draw_weights(4, 2, unit_normal);
    Eigen::RowVectorXd b_1 = draw_weights(4, 1, unit_normal).transpose();
    Eigen::MatrixXd W_2 = draw_weights(1, 4, unit_normal);
    Eigen::RowVectorXd b_2 = draw_weights(1, 1, unit_normal).transpose();

    // std::cout << "Random Weights Matrix: \n" << W_1 << std::endl;

    for (int i = 0; i < kEpochs; ++i) {
        auto [x, y] = draw_sample(kSampleSize, rand_normal);

        // forward pass
        Eigen::MatrixXd a = (x * W_1.transpose()).rowwise() + b_1;
        auto y_1 = relu(a);
        auto y_hat = sigmoid((y_1 * W_2.transpose()).rowwise() + b_2);

        // loss
        auto err = y_hat - y;
        Eigen::MatrixXd pred = (y_hat.array() > 0.5).cast<double>();
        double class_err_rate = (y - pred).array().abs().mean();
        double train_err = err.array().square().mean();

        // backward pass
        Eigen::MatrixXd dW_2 = (err.transpose() * y_1) / y_1.rows();
        Eigen::VectorXd db_2(1);
        db_2 << err.mean();

        Eigen::MatrixXd dy_1 = err * W_2;
        Eigen::MatrixXd mask = (a.array() > 0).cast<double>();
        Eigen::MatrixXd da = dy_1.array() * mask.array(); // relu derivative

        Eigen::MatrixXd dW_1 = (da.transpose() * x) / x.rows();
        Eigen::RowVectorXd db_1 = da.colwise().mean();

        W_1 -= kLearningRate * dW_1;
        b_1 -= kLearningRate * db_1;

        W_2 -= kLearningRate * dW_2;
        b_2 -= kLearningRate * db_2;

        // logging
        if (i % 10 == 0) {
            std::cout << "epoch = " << i
                    << "  train_err = " << train_err
                    << "  class error rate = " << class_err_rate << std::endl;
            std::cout << "   dW_1.norm() " << dW_1.norm() << std::endl;
            std::cout << "   db_1.norm() " << db_1.norm() << std::endl;
            std::cout << "   dW_2.norm() " << dW_2.norm() << std::endl;
            std::cout << "   db_2.norm() " << db_2.norm() << std::endl;

            std::cout << std::endl;
        }
    }
}


int main() {
    neural_networks_example();

    return 0;
}
