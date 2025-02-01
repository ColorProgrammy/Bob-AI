#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <numeric>

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

int main() {
    // Создаем нейронную сеть с 2 входами
    SimpleNeuralNetwork nn(2);

    // Обучающие данные (чашки кофе, часы сна, уровень бодрости)
    std::vector<std::vector<double>> inputs = {
        {0, 8}, // 0 чашек, 8 часов сна
        {1, 6}, // 1 чашка, 6 часов сна
        {2, 4}, // 2 чашки, 4 часа сна
        {0, 4}, // 0 чашек, 4 часа сна
        {1, 8}, // 1 чашка, 8 часов сна
         {2, 6}  // 2 чашки, 6 часов сна
    };
    std::vector<double> targets = {
        0.5, // не очень бодр
        0.6, // более менее бодр
        0.8, // бодр
        0.2,  // не бодр
        0.8, // бодр
        0.9, // очень бодр
    };


    // Обучаем нейросеть
    nn.train(inputs, targets, 0.2, 10000);


    // Тестирование
    std::cout << "Testing:" << std::endl;
      for (size_t i = 0; i < inputs.size(); ++i) {
          double output = nn.forward(inputs[i]);
          std::cout << "Coffees: " << inputs[i][0] << ", Sleep: " << inputs[i][1]
                 << " ->  Bodrost: " << output  << " (Target: " << targets[i] << ")" << std::endl;
       }

       // Тест нового значения
        std::vector<double> test_input = {1, 5};
        double output = nn.forward(test_input);
         std::cout << "Test:  Coffees: " << test_input[0] << ", Sleep: " << test_input[1]
                 << " ->  Bodrost: " << output << std::endl;


    return 0;
}
