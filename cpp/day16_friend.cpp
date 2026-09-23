#include <iostream>

// class Tensor {
// private:
//     int size_;

// public:
//     explicit Tensor(int size) : size_(size) {}

//     friend void print_tensor(const Tensor& t);
// };

// void print_tensor(const Tensor& t) {
//     std::cout << "size = " << t.size_ << '\n';
// }

class Tensor {
private:
    int size_ = 128;

    friend class TensorDebugger;
};

class TensorDebugger {
public:
    void inspect(const Tensor& t) {
        std::cout << t.size_ << '\n';
    }
};

int main() {
    Tensor t(128);
    print_tensor(t);
}