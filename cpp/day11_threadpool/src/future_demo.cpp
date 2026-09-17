#include <iostream>
#include <future>
#include <thread>
#include <chrono>

int compute()
{
    std::this_thread::sleep_for(
        std::chrono::seconds(2)
    );

    return 42;
}

int main()
{
    std::future<int> result =
        std::async(std::launch::async, compute);

    std::cout << "Task submitted" << std::endl;

    int value = result.get();

    std::cout << "Result = " << value << std::endl;

    return 0;
}