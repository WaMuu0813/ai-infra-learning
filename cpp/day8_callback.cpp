#include <iostream>
#include <functional>

int add(int a, int b) {
    return a + b;
}

int multiply(int a, int b) {
    return a * b;
}

int calculate(
    int a,
    int b,
    // int (*operation)(int, int)
    const std::function<int(int, int)>& operation
) {
    return operation(a, b);
}

int main() {
    // std::cout << calculate(3, 5, add) << '\n';
    // std::cout << calculate(3, 5, multiply) << '\n';
    
    int bias = 10;

    auto add_bias = [bias](int a, int b) {
        return a + b + bias;
    };
    
    std::cout << calculate(3, 5, add_bias) << '\n';


    return 0;
}