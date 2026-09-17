#include <iostream>
#include <chrono>
#include <thread>

#include "thread_pool.h"


int main()
{
    ThreadPool pool(4);


    // for(int i = 0; i < 10; i++)
    // {
    //     pool.enqueue(
    //         [i]()
    //         {
    //             std::cout 
    //                 << "Task "
    //                 << i
    //                 << " is running in thread "
    //                 << std::this_thread::get_id()
    //                 << std::endl;


    //             std::this_thread::sleep_for(
    //                 std::chrono::seconds(1)
    //             );


    //             std::cout
    //                 << "Task "
    //                 << i
    //                 << " finished"
    //                 << std::endl;
    //         }
    //     );
    // }


    // std::this_thread::sleep_for(
    //     std::chrono::seconds(5)
    // );

    // 任务1：返回 int
    auto int_result = pool.enqueue(
        []()
        {
            std::this_thread::sleep_for(
                std::chrono::seconds(2)
            );

            return 42;
        }
    );

    // 任务2：返回 std::string
    auto string_result = pool.enqueue(
        []()
        {
            std::this_thread::sleep_for(
                std::chrono::seconds(1)
            );

            return std::string("AI Infra");
        }
    );

    std::cout << "Tasks submitted" << std::endl;

    std::cout
        << "int result = "
        << int_result.get()
        << std::endl;

    std::cout
        << "string result = "
        << string_result.get()
        << std::endl;


    return 0;
}