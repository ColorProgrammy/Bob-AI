#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <numeric>
#include <algorithm>
#include <map>
#include <string>
#include <limits>

// Leaky ReLU функция активации
double leaky_relu(double x, double alpha = 0.01)
{
    return x > 0 ? x : alpha * x;
}

// Производная Leaky ReLU
double leaky_relu_derivative(double x, double alpha = 0.01)
{
    return x > 0 ? 1.0 : alpha;
}

std::vector<double> softmax(const std::vector<double>& inputs) {
    std::vector<double> output;

    if (inputs.empty()) {
        return output;
    }

    // Проверка на особые значения во входных данных
    for (double val : inputs) {
        if (!std::isfinite(val)) {
            return std::vector<double>(inputs.size(), std::numeric_limits<double>::quiet_NaN());
        }
    }

    const double max_val = *std::max_element(inputs.begin(), inputs.end());
    output.reserve(inputs.size());
    double sum_exp = 0.0;

    // Вычисление экспонент с защитой от переполнения
    for (double val : inputs) {
        const double clipped_val = val - max_val;
        const double exp_val = std::exp(clipped_val);
        
        output.push_back(exp_val);
        sum_exp += exp_val;
    }

    // Дополнительные проверки для sum_exp
    if (sum_exp <= 0 || !std::isfinite(sum_exp)) {
        const double uniform_prob = 1.0 / inputs.size();
        return std::vector<double>(inputs.size(), uniform_prob);
    }

    // Нормализация с защитой от деления на ноль
    for (auto& val : output) {
        val /= sum_exp;
        
        // Защита от NaN после нормализации
        if (!std::isfinite(val)) {
            val = 0.0;
        }
    }

    // Дополнительная проверка: нормализация до 1
    const double total = std::accumulate(output.begin(), output.end(), 0.0);
    if (std::abs(total - 1.0) > 1e-6) {
        std::fill(output.begin(), output.end(), 1.0 / output.size());
    }

    return output;
}

class RNN
{
  public:
    RNN(int input_size, int hidden_size, int output_size)
        : input_size_(input_size), hidden_size_(hidden_size), output_size_(output_size)
    {
        // Инициализация весов и смещений случайными числами
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> distrib(-0.1, 0.1);

        weights_ih_.resize(hidden_size_, std::vector<double>(input_size_));
        for (int i = 0; i < hidden_size_; i++)
        {
            for (int j = 0; j < input_size_; j++)
            {
                weights_ih_[i][j] = distrib(gen);
            }
        }

        weights_hh_.resize(hidden_size_, std::vector<double>(hidden_size_));
        for (int i = 0; i < hidden_size_; i++)
        {
            for (int j = 0; j < hidden_size_; j++)
            {
                weights_hh_[i][j] = distrib(gen);
            }
        }

        biases_h_.resize(hidden_size_);
        for (int i = 0; i < hidden_size_; ++i)
        {
            biases_h_[i] = distrib(gen);
        }

        weights_ho_.resize(output_size_, std::vector<double>(hidden_size_));
        for (int i = 0; i < output_size_; i++)
        {
            for (int j = 0; j < hidden_size_; j++)
            {
                weights_ho_[i][j] = distrib(gen);
            }
        }

        biases_o_.resize(output_size_);
        for (int i = 0; i < output_size_; ++i)
        {
            biases_o_[i] = distrib(gen);
        }

        hidden_state_.resize(hidden_size_);
    }

    // Функция прямого распространения
    std::vector<double> forward(const std::vector<double> &inputs)
    {
        std::vector<double> new_hidden_state(hidden_size_);
        for (int i = 0; i < hidden_size_; ++i)
        {
            double weighted_sum = 0.0;
            for (int j = 0; j < input_size_; ++j)
            {
                weighted_sum += inputs[j] * weights_ih_[i][j];
            }
            for (int j = 0; j < hidden_size_; ++j)
            {
                weighted_sum += hidden_state_[j] * weights_hh_[i][j];
            }
            weighted_sum += biases_h_[i];
            new_hidden_state[i] = leaky_relu(weighted_sum);
        }
        hidden_state_ = new_hidden_state;

        std::vector<double> output(output_size_);
        for (int i = 0; i < output_size_; ++i)
        {
            double weighted_sum = 0.0;
            for (int j = 0; j < hidden_size_; j++)
            {
                weighted_sum += hidden_state_[j] * weights_ho_[i][j];
            }
            weighted_sum += biases_o_[i];
            output[i] = weighted_sum;
        }
        return softmax(output);
    }

    void reset_hidden_state()
    {
        std::fill(hidden_state_.begin(), hidden_state_.end(), 0.0);
    }

    // Функция обучения
    void train(const std::vector<std::vector<double>> &inputs,
               const std::vector<int> &targets,
               double learning_rate,
               int epochs)
    {
        std::vector<double> m_ih, v_ih, m_hh, v_hh, m_h, v_h, m_ho, v_ho, m_o, v_o;
        for (int i = 0; i < hidden_size_; i++)
        {
            for (int j = 0; j < input_size_; j++)
            {
                m_ih.push_back(0.0);
                v_ih.push_back(0.0);
            }
        }
        for (int i = 0; i < hidden_size_; i++)
        {
            for (int j = 0; j < hidden_size_; j++)
            {
                m_hh.push_back(0.0);
                v_hh.push_back(0.0);
            }
        }
        for (int i = 0; i < hidden_size_; ++i)
        {
            m_h.push_back(0.0);
            v_h.push_back(0.0);
        }
        for (int i = 0; i < output_size_; i++)
        {
            for (int j = 0; j < hidden_size_; j++)
            {
                m_ho.push_back(0.0);
                v_ho.push_back(0.0);
            }
        }
        for (int i = 0; i < output_size_; ++i)
        {
            m_o.push_back(0.0);
            v_o.push_back(0.0);
        }

        for (int epoch = 0; epoch < epochs; ++epoch)
        {
            double total_error = 0.0;
            for (size_t i = 0; i < inputs.size(); ++i)
            {
                std::vector<double> output = forward(inputs[i]);
                double error = calculate_cross_entropy_error(output, targets[i]);
                total_error += error;
                backpropagation(inputs[i], output, targets[i], learning_rate, m_ih, v_ih, m_hh, v_hh, m_h, v_h, m_ho, v_ho, m_o, v_o, epoch + 1);
            }
            if (epoch % 100 == 0)
            {
                std::cout << "Epoch: " << epoch << ", Error: " << total_error / inputs.size() << std::endl;
            }
        }
    }

    // Функция для подсчета ошибки (кросс-энтропия)
    double calculate_cross_entropy_error(const std::vector<double> &output, int target_index)
    {
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

    void backpropagation(const std::vector<double> &inputs,
                         const std::vector<double> &output,
                         int target_index,
                         double learning_rate,
                         std::vector<double> &m_ih,
                         std::vector<double> &v_ih,
                         std::vector<double> &m_hh,
                         std::vector<double> &v_hh,
                         std::vector<double> &m_h,
                         std::vector<double> &v_h,
                         std::vector<double> &m_ho,
                         std::vector<double> &v_ho,
                         std::vector<double> &m_o,
                         std::vector<double> &v_o,
                         int t)
    {
        double beta1 = 0.9;
        double beta2 = 0.999;
        double epsilon = 1e-8;

        // Выходной слой
        std::vector<double> delta_o(output_size_);
        for (int i = 0; i < output_size_; i++)
        {
            delta_o[i] = output[i];
        }
        delta_o[target_index] -= 1.0;

        for (int i = 0; i < output_size_; i++)
        {
            for (int j = 0; j < hidden_size_; j++)
            {
                int index = i * hidden_size_ + j;
                double grad = delta_o[i] * hidden_state_[j];
                m_ho[index] = beta1 * m_ho[index] + (1 - beta1) * grad;
                v_ho[index] = beta2 * v_ho[index] + (1 - beta2) * grad * grad;
                double m_hat = m_ho[index] / (1 - pow(beta1, t));
                double v_hat = v_ho[index] / (1 - pow(beta2, t));
                weights_ho_[i][j] -= learning_rate * m_hat / (sqrt(v_hat) + epsilon);
            }
            m_o[i] = beta1 * m_o[i] + (1 - beta1) * delta_o[i];
            v_o[i] = beta2 * v_o[i] + (1 - beta2) * delta_o[i] * delta_o[i];
            double m_hat = m_o[i] / (1 - pow(beta1, t));
            double v_hat = v_o[i] / (1 - pow(beta2, t));
            biases_o_[i] -= learning_rate * m_hat / (sqrt(v_hat) + epsilon);
        }

        // Скрытый слой
        std::vector<double> hidden_errors(hidden_size_);
        for (int j = 0; j < hidden_size_; j++)
        {
            double sum = 0.0;
            for (int i = 0; i < output_size_; i++)
            {
                sum += delta_o[i] * weights_ho_[i][j];
            }
            hidden_errors[j] = sum;
        }

        std::vector<double> hidden_deltas(hidden_size_);
        for (int j = 0; j < hidden_size_; j++)
        {
            hidden_deltas[j] = hidden_errors[j] * leaky_relu_derivative(hidden_state_[j]);
        }

        for (int i = 0; i < hidden_size_; i++)
        {
            for (int j = 0; j < input_size_; j++)
            {
                int index = i * input_size_ + j;
                double grad = hidden_deltas[i] * inputs[j];
                m_ih[index] = beta1 * m_ih[index] + (1 - beta1) * grad;
                v_ih[index] = beta2 * v_ih[index] + (1 - beta2) * grad * grad;
                double m_hat = m_ih[index] / (1 - pow(beta1, t));
                double v_hat = v_ih[index] / (1 - pow(beta2, t));
                weights_ih_[i][j] -= learning_rate * m_hat / (sqrt(v_hat) + epsilon);
            }
            for (int j = 0; j < hidden_size_; j++)
            {
                int index = i * hidden_size_ + j;
                double grad = hidden_deltas[i] * hidden_state_[j];
                m_hh[index] = beta1 * m_hh[index] + (1 - beta1) * grad;
                v_hh[index] = beta2 * v_hh[index] + (1 - beta2) * grad * grad;
                double m_hat = m_hh[index] / (1 - pow(beta1, t));
                double v_hat = v_hh[index] / (1 - pow(beta2, t));
                weights_hh_[i][j] -= learning_rate * m_hat / (sqrt(v_hat) + epsilon);
            }
            m_h[i] = beta1 * m_h[i] + (1 - beta1) * hidden_deltas[i];
            v_h[i] = beta2 * v_h[i] + (1 - beta2) * hidden_deltas[i] * hidden_deltas[i];
            double m_hat = m_h[i] / (1 - pow(beta1, t));
            double v_hat = v_h[i] / (1 - pow(beta2, t));
            biases_h_[i] -= learning_rate * m_hat / (sqrt(v_hat) + epsilon);
        }
    }
};

// Функция для преобразования символов в one-hot векторы
std::vector<double> one_hot_encode(char c, const std::map<char, int> &char_map)
{
    int index = char_map.at(c);
    int size = char_map.size();
    std::vector<double> encoding(size, 0.0);
    encoding[index] = 1.0;
    return encoding;
}

// Функция для создания словаря символов
std::map<char, int> create_char_map(const std::string &text)
{
    std::map<char, int> char_map;
    int index = 0;
    for (char c : text)
    {
        if (char_map.find(c) == char_map.end())
        {
            char_map[c] = index++;
        }
    }
    return char_map;
}

int main()
{
    std::string text = "The main character is Woody!";
    std::map<char, int> char_map = create_char_map(text);
    int vocab_size = char_map.size();

    RNN rnn(vocab_size, 128, vocab_size); // input, hidden, output sizes

    std::vector<std::vector<double>> inputs;
    std::vector<int> targets;

    // Создание обучающих данных
    for (size_t i = 0; i < text.size() - 1; ++i)
    {
        inputs.push_back(one_hot_encode(text[i], char_map));
        targets.push_back(char_map.at(text[i + 1]));
    }

    // Обучение нейросети
    rnn.train(inputs, targets, 0.0005, 101);

    rnn.reset_hidden_state();
    char start_char = text[0];
    std::vector<double> current_input = one_hot_encode(start_char, char_map);
    std::string generated_text;
    generated_text += start_char;

    // Тестирование на обучающих данных
    std::cout << "\nTesting on train dataset:" << std::endl;

    std::vector<double> first_input = inputs.empty() ? std::vector<double>(vocab_size, 0.0) : inputs[0];

    char first_char = ' ';
    for (const auto &pair : char_map)
    {
        if (pair.second == std::distance(first_input.begin(), std::max_element(first_input.begin(), first_input.end())))
        {
            first_char = pair.first;
            break;
        }
    }

    std::cout << "Input: ";
    for (const auto &value : first_input)
    {
        std::cout << value << " ";
    }

    char target_char_first = ' ';

    for (const auto &pair : char_map)
    {
        if (pair.second == std::distance(first_input.begin(), std::max_element(first_input.begin(), first_input.end())))
        {
            target_char_first = pair.first;
            break;
        }
    }

    std::cout << "Predicted: " << first_char << ", Expected: " << target_char_first << std::endl;

    for (size_t i = 0; i < inputs.size(); i++)
    {
        std::vector<double> output = rnn.forward(inputs[i]);

        int predicted_index = std::distance(output.begin(), std::max_element(output.begin(), output.end()));
        char predicted_char = ' ';
        for (const auto &pair : char_map)
        {
            if (pair.second == predicted_index)
            {
                predicted_char = pair.first;
                break;
            }
        }

        char target_char = ' ';

        if (i < targets.size())
        {
            for (const auto &pair : char_map)
            {
                if (pair.second == targets[i])
                {
                    target_char = pair.first;
                    break;
                }
            }
        }

        std::cout << "Input: ";
        for (const auto &value : inputs[i])
        {
            std::cout << value << " ";
        }
        std::cout << "Predicted: " << predicted_char << ", Expected: " << target_char << std::endl;

        generated_text += predicted_char;
        current_input = one_hot_encode(predicted_char, char_map);
    }

    std::cout << "\nConcevied: " << text;

    std::cout << "\nGenerated text: " << generated_text << std::endl;

    if (text == generated_text)
    {
        std::cout << "\nBob: As far as I understand, I learned this word!";
    }
    else
    {
        std::cout << "\nBob: What a mess it turned out to be: \"" << generated_text << "\". Uh...";
    }
    return 0;
}
