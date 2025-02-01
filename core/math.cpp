#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <numeric>
#include <algorithm>

// Функция активации (сигмоид)
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

// Производная сигмоида
double sigmoid_derivative(double x) {
    double sigm = sigmoid(x);
    return sigm * (1 - sigm);
}

// ReLU функция активации
double relu(double x) {
    return std::max(0.0, x);
}

// Производная ReLU
double relu_derivative(double x) {
    return x > 0.0 ? 1.0 : 0.0;
}

class NeuralNetwork {
public:
    NeuralNetwork(int input_size, int hidden_size)
        : input_size_(input_size), hidden_size_(hidden_size) {
        // Инициализация весов и смещений случайными числами
        std::random_device rd;
        std::mt19937 gen(rd());
         std::uniform_real_distribution<> distrib(-0.1, 0.1);

        weights_ih_.resize(hidden_size_, std::vector<double>(input_size_));
         for(int i = 0; i < hidden_size_; i++) {
              for(int j = 0; j < input_size_; j++){
               weights_ih_[i][j] = distrib(gen);
            }
         }

        biases_h_.resize(hidden_size_);
        for (int i = 0; i < hidden_size_; ++i) {
            biases_h_[i] = distrib(gen);
        }
        weights_ho_.resize(hidden_size_);
         for(int i = 0; i < hidden_size_; i++) {
           weights_ho_[i] = distrib(gen);
        }

        bias_o_ = distrib(gen);
    }

    // Функция прямого распространения
    double forward(const std::vector<double>& inputs) {
        hidden_outputs_.resize(hidden_size_);
        for (int i = 0; i < hidden_size_; ++i) {
            double weighted_sum = 0.0;
            for (int j = 0; j < input_size_; ++j) {
                weighted_sum += inputs[j] * weights_ih_[i][j];
            }
            weighted_sum += biases_h_[i];
            hidden_outputs_[i] = relu(weighted_sum);
        }

        double output = 0;
           for(int i = 0; i < hidden_size_; ++i) {
            output += hidden_outputs_[i] * weights_ho_[i];
           }
        output += bias_o_;
         return output;
    }

    // Функция обучения
    void train(const std::vector<std::vector<double>>& inputs,
        const std::vector<double>& targets,
        double learning_rate,
        int epochs) {
        for (int epoch = 0; epoch < epochs; ++epoch) {
           double total_error = 0.0;
            for (size_t i = 0; i < inputs.size(); ++i) {
                double output = forward(inputs[i]);
                 double error = targets[i] - output;
                total_error += error * error;
                 backpropagation(inputs[i], error, learning_rate);
            }
            if(epoch % 100 == 0) {
                 std::cout << "Epoch: " << epoch << ", Error: " << total_error / inputs.size() << std::endl;
             }
        }
    }

private:
    int input_size_;
    int hidden_size_;

    std::vector<std::vector<double>> weights_ih_;
    std::vector<double> biases_h_;
    std::vector<double> weights_ho_;
    double bias_o_;
    std::vector<double> hidden_outputs_;

    void backpropagation(const std::vector<double>& inputs,
        double error,
        double learning_rate) {
            double delta_o = error;
           for(int i = 0; i < hidden_size_; i++) {
             weights_ho_[i] += learning_rate * delta_o * hidden_outputs_[i];
           }
           bias_o_ += learning_rate * delta_o;


           std::vector<double> hidden_errors(hidden_size_);
            for(int j = 0; j < hidden_size_; j++) {
              hidden_errors[j] = delta_o * weights_ho_[j];
            }

           std::vector<double> hidden_deltas(hidden_size_);
           for(int j = 0; j < hidden_size_; j++) {
             hidden_deltas[j] = hidden_errors[j] * relu_derivative(hidden_outputs_[j]);
           }


            for(int i = 0; i < hidden_size_; i++) {
                for(int j = 0; j < input_size_; j++) {
                 weights_ih_[i][j] += learning_rate * hidden_deltas[i] * inputs[j];
               }
               biases_h_[i] += learning_rate * hidden_deltas[i];
           }

    }

};

int main() {
    NeuralNetwork nn(1, 32); // 1 вход, 32 нейрона в скрытом слое

    std::vector<std::vector<double>> inputs;
    std::vector<double> targets;

    // Создание обучающих данных
    for (double x = 0; x < 2 * M_PI; x += 0.1) {
        inputs.push_back({x});
        targets.push_back(sin(x));
    }

    // Обучение нейросети
    nn.train(inputs, targets, 0.001, 20000);

      std::cout << "Testing on train dataset:" << std::endl;
         for (size_t i = 0; i < 10; ++i) {
             double predicted = nn.forward(inputs[i]);
           std::cout << "Input: " << inputs[i][0]
                     << " Predicted: " << predicted
                    << ", Expected: " << targets[i] << std::endl;
         }

    // Тестирование на новом значении
    double test_x = 1.5;
    double predicted = nn.forward({test_x});
    std::cout << "\nTesting on a new example: " << std::endl;
    std::cout << "Input: " << test_x << ", Predicted: " << predicted << ", Expected: " << sin(test_x) << std::endl;


    return 0;
}
