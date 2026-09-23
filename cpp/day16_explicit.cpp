#include <iostream>

class Tensor {
private:
    int size_;

public:
    Tensor(int size) : size_(size) {
        std::cout << "Tensor(int), size = " << size_ << '\n';
    }

    int size() const {
        return size_;
    }
};

void process(const Tensor& t) {
    std::cout << "process: " << t.size() << '\n';
}

int main() {
    Tensor a(128);      // ①
    Tensor b = 256;     // ②

    process(a);         // ③
    process(512);       // ④
}