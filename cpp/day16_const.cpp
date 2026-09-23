#include <iostream>

class Tensor {
private:
    int size_;

public:
    explicit Tensor(int size) : size_(size) {}

    void set_size(int size) {
        size_ = size;
    }

    int get_size() {
        return size_;
    }
};

void inspect(const Tensor& t) {
    std::cout << t.get_size() << '\n';
}

int main() {
    Tensor t(128);
    inspect(t);
}