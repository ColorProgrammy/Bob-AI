#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <numeric>
#include <algorithm>
#include <map>

// Сигмоид функция активации
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

// Производная сигмоиды
double sigmoid_derivative(double x) {
    double sigm = sigmoid(x);
    return sigm * (1 - sigm);
}

// Функция softmax для классификации
std::vector<double> softmax(const std::vector<double>& inputs) {
    std::vector<double> output(inputs.size());
    double max_val = *std::max_element(inputs.begin(), inputs.end());
    double sum_exp = 0.0;
   for (size_t i = 0; i < inputs.size(); ++i) {
       double current = exp(inputs[i] - max_val);
       if (current > 1e10) {
           current = 1e10;
       }
       output[i] = current;
      sum_exp += output[i];
    }
    for (size_t i = 0; i < output.size(); ++i) {
        output[i] /= sum_exp;
    }
   return output;
}


class RNN {
public:
    RNN(int input_size, int hidden_size, int output_size)
        : input_size_(input_size), hidden_size_(hidden_size), output_size_(output_size) {
         // Инициализация весов и смещений случайными числами
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> distrib(-0.1, 0.1);

        weights_ih_.resize(hidden_size_, std::vector<double>(input_size_));
        for(int i = 0; i < hidden_size_; i++){
            for(int j = 0; j < input_size_; j++){
                weights_ih_[i][j] = distrib(gen);
            }
        }

        weights_hh_.resize(hidden_size_, std::vector<double>(hidden_size_));
         for(int i = 0; i < hidden_size_; i++){
             for(int j = 0; j < hidden_size_; j++) {
                weights_hh_[i][j] = distrib(gen);
            }
         }


        biases_h_.resize(hidden_size_);
        for (int i = 0; i < hidden_size_; ++i) {
           biases_h_[i] = distrib(gen);
        }

        weights_ho_.resize(output_size_, std::vector<double>(hidden_size_));
        for (int i = 0; i < output_size_; i++) {
            for(int j = 0; j < hidden_size_; j++) {
                weights_ho_[i][j] = distrib(gen);
            }
        }


         biases_o_.resize(output_size_);
         for(int i = 0; i < output_size_; i++) {
             biases_o_[i] = distrib(gen);
          }
         hidden_state_.resize(hidden_size_);
    }

    // Функция прямого распространения
    std::vector<double> forward(const std::vector<double>& inputs) {
       std::vector<double> new_hidden_state(hidden_size_);
        for (int i = 0; i < hidden_size_; ++i) {
            double weighted_sum = 0.0;
            for (int j = 0; j < input_size_; ++j) {
                weighted_sum += inputs[j] * weights_ih_[i][j];
            }
             for (int j = 0; j < hidden_size_; ++j) {
                 weighted_sum += hidden_state_[j] * weights_hh_[i][j];
            }
           weighted_sum += biases_h_[i];
           new_hidden_state[i] = sigmoid(weighted_sum);
        }
         hidden_state_ = new_hidden_state;

         std::vector<double> output(output_size_);
        for (int i = 0; i < output_size_; ++i) {
            double weighted_sum = 0.0;
            for(int j = 0; j < hidden_size_; ++j) {
                weighted_sum += hidden_state_[j] * weights_ho_[i][j];
            }
            weighted_sum += biases_o_[i];
           output[i] = weighted_sum;
        }
       return softmax(output);
    }

    // Функция обучения
    void train(const std::vector<std::vector<double>>& inputs,
               const std::vector<int>& targets,
               double learning_rate,
               int epochs) {
        for (int epoch = 0; epoch < epochs; ++epoch) {
            double total_error = 0.0;
           for (size_t i = 0; i < inputs.size(); ++i) {
                 std::vector<double> output = forward(inputs[i]);
                double error = calculate_cross_entropy_error(output, targets[i]);
                total_error += error;
                backpropagation(inputs[i], output, targets[i], learning_rate);
           }
          if (epoch % 100 == 0) {
              std::cout << "Epoch: " << epoch << ", Error: " << total_error / inputs.size() << std::endl;
         }
      }
  }

    // Функция для подсчета ошибки (кросс-энтропия)
    double calculate_cross_entropy_error(const std::vector<double>& output, int target_index) {
        double error = -log(output[target_index]);
        return error;
    }

private:
    int input_size_;
    int hidden_size_;
    int output_size_;

    std::vector<std::vector<double>> weights_ih_;
    std::vector<std::vector<double>> weights_hh_;
      std::vector<double> biases_h_;
    std::vector<std::vector<double>> weights_ho_;
    std::vector<double> biases_o_;
    std::vector<double> hidden_state_;

    void backpropagation(const std::vector<double>& inputs,
                         const std::vector<double>& output,
                         int target_index,
                        double learning_rate) {

          // Выходной слой
          std::vector<double> delta_o(output_size_);
          for (int i = 0; i < output_size_; i++) {
             delta_o[i] = output[i];
         }
         delta_o[target_index] -= 1.0;

          for (int i = 0; i < output_size_; i++) {
                for (int j = 0; j < hidden_size_; j++) {
                   weights_ho_[i][j] += learning_rate * delta_o[i] * hidden_state_[j];
                }
               biases_o_[i] += learning_rate * delta_o[i];
          }
         // Скрытый слой
        std::vector<double> hidden_errors(hidden_size_);
          for(int j = 0; j < hidden_size_; j++) {
               double sum = 0.0;
               for(int i = 0; i < output_size_; i++) {
                 sum += delta_o[i] * weights_ho_[i][j];
               }
                 hidden_errors[j] = sum;
            }


       std::vector<double> hidden_deltas(hidden_size_);
         for(int j = 0; j < hidden_size_; j++) {
             hidden_deltas[j] = hidden_errors[j] * sigmoid_derivative(hidden_state_[j]);
           }


          for(int i = 0; i < hidden_size_; i++) {
             for(int j = 0; j < input_size_; j++) {
                  weights_ih_[i][j] += learning_rate * hidden_deltas[i] * inputs[j];
              }
             for(int j = 0; j < hidden_size_; j++) {
                  weights_hh_[i][j] += learning_rate * hidden_deltas[i] * hidden_state_[j];
              }
                 biases_h_[i] += learning_rate * hidden_deltas[i];
         }
         normalize_weights();
    }

     void normalize_weights() {
          for (int i = 0; i < hidden_size_; i++) {
            double sum = 0.0;
           for(int j = 0; j < input_size_; j++) {
             sum += weights_ih_[i][j] * weights_ih_[i][j];
            }
           for(int j = 0; j < hidden_size_; j++) {
             sum += weights_hh_[i][j] * weights_hh_[i][j];
             }
            if (sum > 1e-5) {
              sum = sqrt(sum);
             for(int j = 0; j < input_size_; j++) {
                 weights_ih_[i][j] /= sum;
               }
                for(int j = 0; j < hidden_size_; j++) {
                    weights_hh_[i][j] /= sum;
                  }
            }
         }
        for (int i = 0; i < output_size_; i++) {
            double sum = 0.0;
           for (int j = 0; j < hidden_size_; j++) {
              sum += weights_ho_[i][j] * weights_ho_[i][j];
           }
          if (sum > 1e-5) {
               sum = sqrt(sum);
              for (int j = 0; j < hidden_size_; j++) {
                  weights_ho_[i][j] /= sum;
               }
           }
       }
    }
};

// Функция для преобразования символов в one-hot векторы
std::vector<double> one_hot_encode(char c, const std::map<char, int>& char_map) {
    int index = char_map.at(c);
   int size = char_map.size();
    std::vector<double> encoding(size, 0.0);
    encoding[index] = 1.0;
   return encoding;
}

// Функция для создания словаря символов
std::map<char, int> create_char_map(const std::string& text) {
    std::map<char, int> char_map;
    int index = 0;
     for (char c : text) {
         if(char_map.find(c) == char_map.end()) {
           char_map[c] = index++;
        }
    }
    return char_map;
}

int main() {
    std::string text = "hel";
     std::map<char, int> char_map = create_char_map(text);
    int vocab_size = char_map.size();


    RNN rnn(vocab_size, 2, vocab_size); // input, hidden, output sizes

    std::vector<std::vector<double>> inputs;
    std::vector<int> targets;

   for (size_t i = 0; i < text.size() - 1; ++i) {
        inputs.push_back(one_hot_encode(text[i], char_map));
        targets.push_back(char_map.at(text[i+1]));
    }

    rnn.train(inputs, targets, 0.005, 20000);


    std::cout << "Testing on train dataset:" << std::endl;
    for (size_t i = 0; i < inputs.size(); i++) {
       std::vector<double> output = rnn.forward(inputs[i]);
       int predicted_index = std::distance(output.begin(), std::max_element(output.begin(), output.end()));
       char predicted_char;
       for (const auto& pair : char_map) {
           if (pair.second == predicted_index) {
               predicted_char = pair.first;
                break;
            }
        }
          char target_char;
         for (const auto& pair : char_map) {
            if (pair.second == targets[i]) {
                 target_char = pair.first;
                 break;
            }
         }
        std::cout << "Input: ";
        for (const auto& value : inputs[i]) {
          std::cout << value << " ";
        }
        std::cout << "Predicted: " << predicted_char << ", Expected: " << target_char << std::endl;
   }


    std::cout << "\nGenerating text:" << std::endl;
    std::string generated_text = "";
    char current_char = 'h';
    for(int i = 0; i < 4; i++) {
        std::vector<double> current_input = one_hot_encode(current_char, char_map);
        std::vector<double> output = rnn.forward(current_input);
        int predicted_index = std::distance(output.begin(), std::max_element(output.begin(), output.end()));
        char predicted_char;
         for (const auto& pair : char_map) {
            if (pair.second == predicted_index) {
                 predicted_char = pair.first;
                break;
           }
        }
       generated_text += predicted_char;
       current_char = predicted_char;
    }

    std::cout << "Generated text: " << generated_text << std::endl;

    return 0;
}
