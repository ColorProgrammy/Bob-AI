#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
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

class NeuralNetwork {
public:
    NeuralNetwork(int input_size, int hidden_size, int output_size)
        : input_size_(input_size), hidden_size_(hidden_size), output_size_(output_size) {
        // Инициализация весов и смещений случайными числами
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> distrib(-1.0, 1.0);

        weights_ih_.resize(hidden_size_, std::vector<double>(input_size_));
        for (int i = 0; i < hidden_size_; ++i) {
            for (int j = 0; j < input_size_; ++j) {
                weights_ih_[i][j] = distrib(gen);
            }
        }
        biases_h_.resize(hidden_size_);
        for (int i = 0; i < hidden_size_; ++i) {
            biases_h_[i] = distrib(gen);
        }

        weights_ho_.resize(output_size_, std::vector<double>(hidden_size_));
        for (int i = 0; i < output_size_; ++i) {
            for (int j = 0; j < hidden_size_; ++j) {
                weights_ho_[i][j] = distrib(gen);
            }
        }
        biases_o_.resize(output_size_);
        for (int i = 0; i < output_size_; ++i) {
            biases_o_[i] = distrib(gen);
        }
    }

    // Функция прямого распространения
    std::vector<double> forward(const std::vector<double>& inputs) {
        hidden_outputs_.resize(hidden_size_);
        for (int i = 0; i < hidden_size_; ++i) {
            double weighted_sum = 0.0;
            for (int j = 0; j < input_size_; ++j) {
                weighted_sum += inputs[j] * weights_ih_[i][j];
            }
            weighted_sum += biases_h_[i];
            hidden_outputs_[i] = sigmoid(weighted_sum);
        }

        output_outputs_.resize(output_size_);
        for (int i = 0; i < output_size_; ++i) {
            double weighted_sum = 0.0;
            for (int j = 0; j < hidden_size_; ++j) {
                weighted_sum += hidden_outputs_[j] * weights_ho_[i][j];
            }
            weighted_sum += biases_o_[i];
            output_outputs_[i] = sigmoid(weighted_sum);
        }

        return output_outputs_;
    }

    // Функция обучения нейросети
    void train(const std::vector<std::vector<double>>& inputs,
        const std::vector<std::vector<double>>& targets,
        double learning_rate,
        int epochs) {
        for (int epoch = 0; epoch < epochs; ++epoch) {
            double total_error = 0.0;
            for (size_t i = 0; i < inputs.size(); ++i) {
                std::vector<double> outputs = forward(inputs[i]);
                std::vector<double> errors(output_size_);
                for (int j = 0; j < output_size_; j++) {
                    errors[j] = targets[i][j] - outputs[j];
                }
                total_error += std::accumulate(errors.begin(), errors.end(), 0.0, [](double sum, double val) {
                        return sum + val * val;
                  });
                  
                backpropagation(inputs[i], errors, learning_rate);
            }
             if (epoch % 100 == 0) {
                std::cout << "Epoch: " << epoch << ", Error: " << total_error / inputs.size() << std::endl;
             }
        }
    }


private:
    int input_size_;
    int hidden_size_;
    int output_size_;

    std::vector<std::vector<double>> weights_ih_; // Веса между входным и скрытым слоями
    std::vector<double> biases_h_; // Смещения скрытого слоя

    std::vector<std::vector<double>> weights_ho_; // Веса между скрытым и выходным слоями
    std::vector<double> biases_o_; // Смещения выходного слоя

    std::vector<double> hidden_outputs_;
    std::vector<double> output_outputs_;

    // Функция обратного распространения
    void backpropagation(const std::vector<double>& inputs,
                         const std::vector<double>& errors,
                         double learning_rate) {

         std::vector<double> output_deltas(output_size_);
         for(int i = 0; i < output_size_; i++) {
           output_deltas[i] = errors[i] * sigmoid_derivative(output_outputs_[i]);
          }
      
        for (int i = 0; i < output_size_; ++i) {
            for (int j = 0; j < hidden_size_; ++j) {
               weights_ho_[i][j] += learning_rate * output_deltas[i] * hidden_outputs_[j];
            }
              biases_o_[i] += learning_rate * output_deltas[i];
        }
        
        std::vector<double> hidden_errors(hidden_size_, 0.0);
        for(int j = 0; j < hidden_size_; j++) {
             for(int k = 0; k < output_size_; k++) {
               hidden_errors[j] += output_deltas[k] * weights_ho_[k][j];
             }
        }
        std::vector<double> hidden_deltas(hidden_size_);
         for (int j = 0; j < hidden_size_; ++j) {
           hidden_deltas[j] = hidden_errors[j] * sigmoid_derivative(hidden_outputs_[j]);
        }


        for (int i = 0; i < hidden_size_; ++i) {
            for (int j = 0; j < input_size_; ++j) {
                weights_ih_[i][j] += learning_rate * hidden_deltas[i] * inputs[j];
            }
            biases_h_[i] += learning_rate * hidden_deltas[i];
        }
    }
};

int main() {
    // Пример использования нейросети для классификации
    NeuralNetwork nn(2, 4, 2); // 2 входа, 4 нейрона в скрытом слое, 2 выхода

    // Учебные данные:
    //  0 - Класс A
    //  1 - Класс B
    std::vector<std::vector<double>> inputs = {
        {0, 0}, {0, 1}, {1, 0}, {1, 1},
         {2, 1}, {1, 2}, {2, 2}, {3, 3}
    };

    std::vector<std::vector<double>> targets = {
        {1, 0}, // 0,0 -> Class A
        {1, 0}, // 0,1 -> Class A
        {1, 0}, // 1,0 -> Class A
        {1, 0},  // 1,1 -> Class A
        {0, 1}, // 2,1 -> Class B
        {0, 1},  // 1,2 -> Class B
         {0, 1}, // 2,2 -> Class B
        {0, 1} // 3,3 -> Class B
    };

    nn.train(inputs, targets, 0.2, 10000); // Обучение

    // Тестирование
    std::cout << "Testing:" << std::endl;
    for (size_t i = 0; i < inputs.size(); i++) {
        std::vector<double> output = nn.forward(inputs[i]);
        int class_index = std::distance(output.begin(), std::max_element(output.begin(), output.end()));
        std::cout << "Input: " << inputs[i][0] << ", " << inputs[i][1] << " -> Class: " << (class_index == 0 ? 'A' : 'B') << std::endl;
    }

    // Тест нового значения
     std::vector<double> test_input = {4,4};
        std::vector<double> test_output = nn.forward(test_input);
         int class_index = std::distance(test_output.begin(), std::max_element(test_output.begin(), test_output.end()));
        std::cout << "Input: " << test_input[0] << ", " << test_input[1] << " -> Class: " << (class_index == 0 ? 'A' : 'B') << std::endl;

    return 0;
}
