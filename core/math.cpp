#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <numeric>
#include <sstream>


// Функция активации (сигмоид)
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

// Производная сигмоида
double sigmoid_derivative(double x) {
    double sigm = sigmoid(x);
    return sigm * (1 - sigm);
}

class SimpleNeuralNetwork {
public:
     SimpleNeuralNetwork(int input_size) : input_size_(input_size) {
        // Инициализация весов и смещения случайными числами
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> distrib(-1.0, 1.0);

        weights_.resize(input_size_);
        for (int i = 0; i < input_size_; ++i) {
            weights_[i] = distrib(gen);
        }
         bias_ = distrib(gen);
    }


    // Функция прямого распространения
    double forward(const std::vector<double>& inputs) {
        double weighted_sum = 0.0;
        for (int i = 0; i < input_size_; ++i) {
            weighted_sum += inputs[i] * weights_[i];
        }
         weighted_sum += bias_;
        return sigmoid(weighted_sum);
    }
       // Функция обучения
    void train(const std::vector<std::vector<double>>& inputs,
               const std::vector<double>& targets,
               double learning_rate,
               int epochs) {
          for(int epoch = 0; epoch < epochs; ++epoch) {
               double total_error = 0.0;
               for (size_t i = 0; i < inputs.size(); ++i) {
                   double output = forward(inputs[i]);
                   double error = targets[i] - output;
                  total_error += error * error;
                  backpropagation(inputs[i], error, learning_rate);
               }
           if (epoch % 100 == 0) {
             std::cout << "Epoch: " << epoch << ", Error: " << total_error / inputs.size() << std::endl;
           }
         }
    }
private:
    int input_size_;
    std::vector<double> weights_;
    double bias_;

    void backpropagation(const std::vector<double>& inputs,
                          double error,
                        double learning_rate) {

             double delta = error * sigmoid_derivative(forward(inputs));

            for (int i = 0; i < input_size_; ++i) {
                  weights_[i] += learning_rate * delta * inputs[i];
           }
            bias_ += learning_rate * delta;
        }

};


double calculate_expression(double a, double b, double c, double d) {
    return a + b * c - d;
}

int main() {
     SimpleNeuralNetwork nn(4); // 4 входа

    std::vector<std::vector<double>> inputs;
    std::vector<double> targets;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> distrib(0.0, 10.0);

    // Создание обучающих данных
    for (int i = 0; i < 1000; ++i) {
        double a = distrib(gen);
        double b = distrib(gen);
        double c = distrib(gen);
        double d = distrib(gen);

        inputs.push_back({a, b, c, d});
        targets.push_back(calculate_expression(a, b, c, d));
    }

    // Обучение нейросети
    nn.train(inputs, targets, 0.01, 10000);

    // Тестирование на обучающих данных
      std::cout << "Testing on train dataset:" << std::endl;
        for (size_t i = 0; i < 10; ++i) {
            double predicted = nn.forward(inputs[i]);
            std::cout << "Input: " << inputs[i][0] << " + " << inputs[i][1] << " * " << inputs[i][2] << " - " << inputs[i][3]
                      << " Predicted: " << predicted
                      << ", Expected: " << targets[i] << std::endl;
       }

   // Тестирование на новом примере
    double a_test = distrib(gen);
    double b_test = distrib(gen);
    double c_test = distrib(gen);
    double d_test = distrib(gen);
    std::vector<double> test_input = {a_test, b_test, c_test, d_test};
    double predicted = nn.forward(test_input);
    std::cout << "\nTesting on a new example: " << std::endl;
    std::cout << "Input: " << a_test << " + " << b_test << " * " << c_test << " - " << d_test
                << ", Predicted: " << predicted
                << ", Expected: " << calculate_expression(a_test, b_test, c_test, d_test)
                << std::endl;


    return 0;
}
