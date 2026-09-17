#include <iostream>
#include <future>

int compute()
{
    return 42;
}

int main()
{
    std::packaged_task<int()> task(compute);

    std::future<int> result = task.get_future();

    task();

    std::cout << result.get() << std::endl;

    return 0;
}