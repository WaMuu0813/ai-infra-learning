#include <iostream>

class Tensor {
private:
    int size_;

public:
    static int count;

    explicit Tensor(int size) : size_(size) {
        ++count;
    }

    int size() const {
        return size_;
    }

    static int get_count() {
        return count;
    }
    
};

int Tensor::count = 0;

int main() {
    std::cout << "start: " << Tensor::count << '\n';

    Tensor a(128);
    std::cout << "after a: " << Tensor::count << '\n';

    Tensor b(256);
    std::cout << "after b: " << Tensor::count << '\n';
}