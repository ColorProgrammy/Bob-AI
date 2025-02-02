#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <algorithm>
#include <sstream>
#include <cmath>
#include <random>
#include <numeric>
#include <limits>

double leaky_relu(double x, double alpha = 0.01) {
    return x > 0 ? x : alpha * x;
}

double leaky_relu_derivative(double x, double alpha = 0.01) {
    return x > 0 ? 1.0 : alpha;
}

std::vector<double> softmax(const std::vector<double>& inputs) {
    std::vector<double> output;
    if (inputs.empty()) return output;
    for (double val : inputs) {
        if (!std::isfinite(val)) {
            return std::vector<double>(inputs.size(), std::numeric_limits<double>::quiet_NaN());
        }
    }
    const double max_val = *std::max_element(inputs.begin(), inputs.end());
    output.reserve(inputs.size());
    double sum_exp = 0.0;
    for (double val : inputs) {
        const double clipped_val = val - max_val;
        const double exp_val = std::exp(clipped_val);
        output.push_back(exp_val);
        sum_exp += exp_val;
    }
    if (sum_exp <= 0 || !std::isfinite(sum_exp)) {
        const double uniform_prob = 1.0 / inputs.size();
        return std::vector<double>(inputs.size(), uniform_prob);
    }
    for (auto& val : output) {
        val /= sum_exp;
        if (!std::isfinite(val)) {
            val = 0.0;
        }
    }
    const double total = std::accumulate(output.begin(), output.end(), 0.0);
    if (std::abs(total - 1.0) > 1e-6) {
        std::fill(output.begin(), output.end(), 1.0 / output.size());
    }
    return output;
}

class RNN {
public:
    RNN(int input_size, int hidden_size, int output_size)
        : input_size_(input_size), hidden_size_(hidden_size), output_size_(output_size) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> distrib(-0.1, 0.1);
        weights_ih_.resize(hidden_size_, std::vector<double>(input_size_));
        for (int i = 0; i < hidden_size_; i++) {
            for (int j = 0; j < input_size_; j++) {
                weights_ih_[i][j] = distrib(gen);
            }
        }
        weights_hh_.resize(hidden_size_, std::vector<double>(hidden_size_));
        for (int i = 0; i < hidden_size_; i++) {
            for (int j = 0; j < hidden_size_; j++) {
                weights_hh_[i][j] = distrib(gen);
            }
        }
        biases_h_.resize(hidden_size_);
        for (int i = 0; i < hidden_size_; ++i) {
            biases_h_[i] = distrib(gen);
        }
        weights_ho_.resize(output_size_, std::vector<double>(hidden_size_));
        for (int i = 0; i < output_size_; i++) {
            for (int j = 0; j < hidden_size_; j++) {
                weights_ho_[i][j] = distrib(gen);
            }
        }
        biases_o_.resize(output_size_);
        for (int i = 0; i < output_size_; ++i) {
            biases_o_[i] = distrib(gen);
        }
        hidden_state_.resize(hidden_size_);
    }

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
            new_hidden_state[i] = leaky_relu(weighted_sum);
        }
        hidden_state_ = new_hidden_state;
        std::vector<double> output(output_size_);
        for (int i = 0; i < output_size_; ++i) {
            double weighted_sum = 0.0;
            for (int j = 0; j < hidden_size_; j++) {
                weighted_sum += hidden_state_[j] * weights_ho_[i][j];
            }
            weighted_sum += biases_o_[i];
            output[i] = weighted_sum;
        }
        return softmax(output);
    }

    void reset_hidden_state() {
        std::fill(hidden_state_.begin(), hidden_state_.end(), 0.0);
    }

    void train(const std::vector<std::vector<double>>& inputs, const std::vector<int>& targets, double learning_rate, int epochs) {
        std::vector<double> m_ih, v_ih, m_hh, v_hh, m_h, v_h, m_ho, v_ho, m_o, v_o;
        for (int i = 0; i < hidden_size_; i++) {
            for (int j = 0; j < input_size_; j++) {
                m_ih.push_back(0.0);
                v_ih.push_back(0.0);
            }
        }
        for (int i = 0; i < hidden_size_; i++) {
            for (int j = 0; j < hidden_size_; j++) {
                m_hh.push_back(0.0);
                v_hh.push_back(0.0);
            }
        }
        for (int i = 0; i < hidden_size_; ++i) {
            m_h.push_back(0.0);
            v_h.push_back(0.0);
        }
        for (int i = 0; i < output_size_; i++) {
            for (int j = 0; j < hidden_size_; j++) {
                m_ho.push_back(0.0);
                v_ho.push_back(0.0);
            }
        }
        for (int i = 0; i < output_size_; ++i) {
            m_o.push_back(0.0);
            v_o.push_back(0.0);
        }
        for (int epoch = 0; epoch < epochs; ++epoch) {
            double total_error = 0.0;
            for (size_t i = 0; i < inputs.size(); ++i) {
                std::vector<double> output = forward(inputs[i]);
                double error = calculate_cross_entropy_error(output, targets[i]);
                total_error += error;
                backpropagation(inputs[i], output, targets[i], learning_rate, m_ih, v_ih, m_hh, v_hh, m_h, v_h, m_ho, v_ho, m_o, v_o, epoch + 1);
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
    std::vector<std::vector<double>> weights_ih_;
    std::vector<std::vector<double>> weights_hh_;
    std::vector<double> biases_h_;
    std::vector<std::vector<double>> weights_ho_;
    std::vector<double> biases_o_;
    std::vector<double> hidden_state_;

    double calculate_cross_entropy_error(const std::vector<double>& output, int target_index) {
        double error = -log(output[target_index]);
        return error;
    }

    void backpropagation(const std::vector<double>& inputs, const std::vector<double>& output, int target_index, double learning_rate, std::vector<double>& m_ih, std::vector<double>& v_ih, std::vector<double>& m_hh, std::vector<double>& v_hh, std::vector<double>& m_h, std::vector<double>& v_h, std::vector<double>& m_ho, std::vector<double>& v_ho, std::vector<double>& m_o, std::vector<double>& v_o, int t) {
        double beta1 = 0.9;
        double beta2 = 0.999;
        double epsilon = 1e-8;
        std::vector<double> delta_o(output_size_);
        for (int i = 0; i < output_size_; i++) {
            delta_o[i] = output[i];
        }
        delta_o[target_index] -= 1.0;
        for (int i = 0; i < output_size_; i++) {
            for (int j = 0; j < hidden_size_; j++) {
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
        std::vector<double> hidden_errors(hidden_size_);
        for (int j = 0; j < hidden_size_; j++) {
            double sum = 0.0;
            for (int i = 0; i < output_size_; i++) {
                sum += delta_o[i] * weights_ho_[i][j];
            }
            hidden_errors[j] = sum;
        }
        std::vector<double> hidden_deltas(hidden_size_);
        for (int j = 0; j < hidden_size_; j++) {
            hidden_deltas[j] = hidden_errors[j] * leaky_relu_derivative(hidden_state_[j]);
        }
        for (int i = 0; i < hidden_size_; i++) {
            for (int j = 0; j < input_size_; j++) {
                int index = i * input_size_ + j;
                double grad = hidden_deltas[i] * inputs[j];
                m_ih[index] = beta1 * m_ih[index] + (1 - beta1) * grad;
                v_ih[index] = beta2 * v_ih[index] + (1 - beta2) * grad * grad;
                double m_hat = m_ih[index] / (1 - pow(beta1, t));
                double v_hat = v_ih[index] / (1 - pow(beta2, t));
                weights_ih_[i][j] -= learning_rate * m_hat / (sqrt(v_hat) + epsilon);
            }
            for (int j = 0; j < hidden_size_; j++) {
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

class RestaurantGame {
    enum class Action { SHOW_MENU, ADD_ITEM, SHOW_BILL, UNKNOWN };
    struct MenuItem { std::string name; double price; };
    RNN rnn;
    std::map<std::string, int> vocabulary;
    std::vector<MenuItem> menu;
    std::map<std::string, int> current_order;

public:
    RestaurantGame() : rnn(50, 64, 4) {
        initialize_menu();
        initialize_vocabulary();
        train_ai();
    }

    void run() {
        std::cout << "Bob: Welcome! How can I help you?\n";
        std::string input;
        while (true) {
            std::cout << "\nYou: ";
            std::getline(std::cin, input);
            Action action = predict_action(input);
            process_action(action, input);
            if (action == Action::SHOW_BILL) break;
        }
    }

private:
    void initialize_menu() {
        menu = { {"pizza", 12.5}, {"pasta", 10.0}, {"salad", 8.5}, {"cola", 2.0} };
    }

    void initialize_vocabulary() {
        std::vector<std::string> base_words = { "menu", "order", "bill", "tip", "pizza", "pasta", "salad", "cola" };
        for (size_t i = 0; i < base_words.size(); ++i) {
            vocabulary[base_words[i]] = i;
        }
    }

    std::vector<double> encode_input(const std::string& text) {
        std::vector<double> encoded(50, 0.0);
        std::istringstream iss(text);
        std::string word;
        int count = 0;
        while (iss >> word && count < 10) {
            std::transform(word.begin(), word.end(), word.begin(), ::tolower);
            if (vocabulary.count(word)) {
                encoded[vocabulary[word]] = 1.0;
            }
            count++;
        }
        return encoded;
    }

    Action predict_action(const std::string& input) {
        std::vector<double> encoded = encode_input(input);
        std::vector<double> output = rnn.forward(encoded);
        int pred_idx = std::distance(output.begin(), std::max_element(output.begin(), output.end()));
        return static_cast<Action>(pred_idx);
    }

    void process_action(Action action, const std::string& input) {
        switch (action) {
            case Action::SHOW_MENU: show_menu(); break;
            case Action::ADD_ITEM: add_to_order(input); break;
            case Action::SHOW_BILL: show_bill(); break;
            default: std::cout << "Bob: I didn't understand that.\n";
        }
    }

    void show_menu() {
        std::cout << "Bob: Here's our menu:\n";
        for (const auto& item : menu) {
            std::cout << "- " << item.name << " ($" << item.price << ")\n";
        }
    }

    void add_to_order(const std::string& input) {
        std::istringstream iss(input);
        int quantity = 1;
        std::string item;
        std::vector<std::string> items;
        while (iss >> item) {
            if (isdigit(item[0])) {
                quantity = std::stoi(item);
            } else {
                items.push_back(item);
            }
        }
        for (const auto& menu_item : menu) {
            for (const auto& word : items) {
                if (menu_item.name.find(word) != std::string::npos) {
                    current_order[menu_item.name] += quantity;
                    std::cout << "Bob: Added " << quantity << " x " << menu_item.name << "\n";
                    return;
                }
            }
        }
        std::cout << "Bob: That's not on the menu.\n";
    }

    void show_bill() {
        double total = 0.0;
        std::cout << "\nBob: Your bill:\n";
        for (const auto& [item, qty] : current_order) {
            auto it = std::find_if(menu.begin(), menu.end(), [&](const MenuItem& m) { return m.name == item; });
            if (it != menu.end()) {
                double price = qty * it->price;
                std::cout << qty << " x " << item << " - $" << price << "\n";
                total += price;
            }
        }
        std::cout << "Total: $" << total << "\n";
        ask_for_tip(total);
    }

    void ask_for_tip(double total) {
        std::cout << "Bob: Would you like to leave a tip? (enter percentage): ";
        double tip_percent;
        std::cin >> tip_percent;
        double tip = total * tip_percent / 100;
        std::cout << "Tip: $" << tip << "\n";
        std::cout << "Grand Total: $" << (total + tip) << "\n";
    }

    void train_ai() {
        std::vector<std::string> training_inputs = { "show menu", "order 2 pizzas", "bring salad", "order 3 salads" };

        std::vector<Action> training_outputs = {
            Action::SHOW_MENU,
            Action::ADD_ITEM,
            Action::ADD_ITEM,
            Action::SHOW_BILL
        };

        // Преобразование в входные векторы
        std::vector<std::vector<double>> inputs;
        std::vector<int> targets;
        
        for(size_t i = 0; i < training_inputs.size(); ++i) {
            inputs.push_back(encode_input(training_inputs[i]));
            targets.push_back(static_cast<int>(training_outputs[i]));
        }
        rnn.train(inputs, targets, 0.001, 500);
    }
};

int main() {
    RestaurantGame game;
    game.run();
    return 0;
}
