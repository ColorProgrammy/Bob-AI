#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <sstream>
#include <ctime>
#include <deque>
#include <regex>
#include <random>
#include <fstream>

using namespace std;

class AdvancedChatBot {
    struct Context {
        deque<string> recent_dialogue;
        string user_name = "Friend";
        string last_topic;
        double mood = 0.5;
    };

    struct KnowledgeEntry {
        vector<string> questions;
        vector<string> answers;
        double weight = 1.0;
        time_t last_used = time(0);
    };

    unordered_map<string, KnowledgeEntry> knowledge_base;
    Context context;
    mt19937 rng;
    string data_file = "chatbot_data.txt";

    vector<string> tokenize(const string& text) {
        vector<string> tokens;
        stringstream ss(normalize(text));
        string token;
        while(ss >> token) {
            if(token.size() > 1) {
                tokens.push_back(token);
            }
        }
        return tokens;
    }

    string normalize(string text) const {
        transform(text.begin(), text.end(), text.begin(), ::tolower);
        text = regex_replace(text, regex(R"([^a-zа-яё0-9' ])"), "");
        return text;
    }

    void learn_phrase(const string& question, const string& answer) {
        auto tokens = tokenize(question);
        if(tokens.empty()) return;

        string key = tokens.front();
        auto& entry = knowledge_base[key];

        if(find(entry.questions.begin(), entry.questions.end(), question) == entry.questions.end()) {
            entry.questions.push_back(question);
        }
        entry.answers.push_back(answer);
        entry.last_used = time(0);
        save_knowledge();
    }

    string find_best_match(const string& input) {
        vector<string> tokens = tokenize(input);
        if(tokens.empty()) return "";

        vector<pair<double, string>> candidates;

        for(const auto& [key, entry] : knowledge_base) {
            if(entry.answers.empty()) continue;

            double score = 0.0;
            score += 0.5 * (1.0 - difftime(time(0), entry.last_used)/86400.0);
            
            for(const auto& token : tokens) {
                if(find(entry.questions.begin(), entry.questions.end(), token) != entry.questions.end()) {
                    score += 1.0;
                }
            }
            
            for(const auto& ctx : context.recent_dialogue) {
                if(ctx.find(key) != string::npos) {
                    score += 0.7;
                }
            }

            candidates.emplace_back(score, key);
        }

        if(!candidates.empty()) {
            sort(candidates.rbegin(), candidates.rend());
            const auto& best = knowledge_base[candidates[0].second];
            if(!best.answers.empty()) {
                return best.answers[rng() % best.answers.size()];
            }
        }
        return "";
    }

    void save_knowledge() {
        ofstream file(data_file);
        if(file.is_open()) {
            for(const auto& [key, entry] : knowledge_base) {
                file << "==ENTRY==\n" << key << "\n";
                for(const auto& q : entry.questions) file << q << "\n";
                file << "==ANSWERS==\n";
                for(const auto& a : entry.answers) file << a << "\n";
                file << "\n";
            }
        }
    }

    void load_knowledge() {
        ifstream file(data_file);
        if(file.is_open()) {
            string line;
            KnowledgeEntry current_entry;
            string current_key;
            bool reading_questions = true;
            bool entry_started = false;

            while(getline(file, line)) {
                if(line == "==ENTRY==") {
                    if(entry_started) {
                        knowledge_base[current_key] = current_entry;
                    }
                    entry_started = true;
                    current_entry = KnowledgeEntry();
                    getline(file, current_key);
                }
                else if(line == "==ANSWERS==") {
                    reading_questions = false;
                }
                else if(!line.empty()) {
                    if(reading_questions) {
                        current_entry.questions.push_back(line);
                    } else {
                        current_entry.answers.push_back(line);
                    }
                }
            }
            if(entry_started) {
                knowledge_base[current_key] = current_entry;
            }
        }
    }

    string handle_personal_question(const string& input) {
        static const unordered_map<string, string> responses = {
            {"your name", "I'm an AI assistant. You can call me Neo."},
            {"your age", "I was born in 2023!"},
            {"favorite color", "I like binary colors - #010101!"},
            {"bye", "Goodbye! Our conversation helped me learn!"}
        };

        for(const auto& [key, response] : responses) {
            if(input.find(key) != string::npos) {
                return response;
            }
        }
        return "";
    }

    string evaluate_action(const string& action) {
        static const unordered_map<string, string> evaluations = {
            {"drink 6 cups of coffee", "That's excessive! It might lead to insomnia."},
            {"sleep 2 hours", "That's not enough sleep. You need more rest."},
            {"exercise", "Great! Physical activity is beneficial for health."},
            {"eat fruits", "Good choice! Fruits are healthy."}
        };

        for(const auto& [key, evaluation] : evaluations) {
            if(action.find(key) != string::npos) {
                return evaluation;
            }
        }
        return "I don't have enough information about that action.";
    }

public:
    AdvancedChatBot() : rng(random_device{}()) {
        load_knowledge();
        
        if(knowledge_base.empty()) {
            learn_phrase("hello", "Hi there! How can I help you today?");
            learn_phrase("hi", "Hello! What's on your mind?");
            learn_phrase("how are you", "I'm great! How about you?");
            learn_phrase("thank you", "You're welcome!");
        }
    }

    string respond(const string& user_input) {
        string input = normalize(user_input);
        if(input.empty()) return "Could you please rephrase that?";

        if(context.recent_dialogue.size() > 5) {
            context.recent_dialogue.pop_front();
        }
        context.recent_dialogue.push_back(input);

        if(auto response = handle_personal_question(input); !response.empty()) {
            return response;
        }

        if(input.find("I ") == 0) {
            return evaluate_action(input);
        }

        if(auto response = find_best_match(input); !response.empty()) {
            return response;
        }

        if(context.recent_dialogue.size() >= 2) {
            const auto& prev_question = context.recent_dialogue[context.recent_dialogue.size()-2];
            learn_phrase(prev_question, input);
        }

        vector<string> follow_ups = {
            "Interesting! Tell me more.",
            "How do you feel about that?",
            "What makes you think that?",
            "Could you explain further?",
            "What's your perspective on this?"
        };

        return follow_ups[rng() % follow_ups.size()];
    }
};

int main() {
    AdvancedChatBot bot;
    string input;

    cout << "Bob: Hi! I'm a learning AI. Let's chat! (Type 'bye' to exit)\n";

    while(true) {
        cout << "You: ";
        getline(cin, input);

        string response = bot.respond(input);
        
        if(response.find("Goodbye") != string::npos) {
            cout << "Bob: " << response << endl;
            break;
        }

        cout << "Bob: " << response << endl;
    }

    return 0;
}
