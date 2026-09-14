#include <iostream>
#include <vector>
#include <algorithm>

template <typename T>
T my_max(T a, T b) {
    return a > b ? a : b;
}

int main() {
    std::cout << my_max(3, 5) << '\n';
    std::cout << my_max(3.2, 5.7) << '\n';

    int x = 10;

    auto f = [x]() {
        return x;
    };

    auto g = [&x]() {
        return x;
    };

    x = 30;

    std::cout << "f() = " << f() << '\n';
    std::cout << "g() = " << g() << '\n';

    std::vector<int> nums = {5, 2, 8, 1, 4};

    std::sort(
        nums.begin(),
        nums.end(),
        [](int a, int b) {
            return a > b;
        }
    );

    // std::cout << "ascending: ";
    std::cout << "descending: ";
    for (int x : nums) {
        std::cout << x << ' ';
    }
    std::cout << '\n';

    auto it = nums.begin();

    std::cout << "first = " << *it << '\n';

    ++it;

    std::cout << "second = " << *it << '\n';

    int sum = 0;
    std::for_each(
        nums.begin(),
        nums.end(),
        [&sum](int x){
            sum += x;
        }
    );
    std::cout << sum << std::endl;

    return 0;
}