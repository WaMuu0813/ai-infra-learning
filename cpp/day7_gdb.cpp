#include <iostream>
#include <vector>

int sum_vector(const std::vector<int>& nums) {
    int sum = 0;

    for (int i = 0; i < nums.size(); ++i) {
        sum += nums[i];
    }

    return sum;
}

int main() {
    std::vector<int> data{10, 20, 30, 40};

    int result = sum_vector(data);

    std::cout << "result = " << result << '\n';

    return 0;
}