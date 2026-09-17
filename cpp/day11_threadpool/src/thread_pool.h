#ifndef THREAD_POOL_H
#define THREAD_POOL_H

#include <vector>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <future>


class ThreadPool
{

public:

    ThreadPool(size_t num_threads);

    ~ThreadPool();


    // void enqueue(std::function<void()> task);
    template<typename F>
    auto enqueue(F task) -> std::future<decltype(task())>
    {
        using ReturnType = decltype(task());
    
        auto packaged_task =
            std::make_shared<std::packaged_task<ReturnType()>>(
                std::move(task)
            );
        
        std::future<ReturnType> result =
            packaged_task->get_future();
        
        {
            std::lock_guard<std::mutex> lock(queue_mutex);
        
            tasks.push(
                [packaged_task]()
                {
                    (*packaged_task)();
                }
            );
        }
    
        condition.notify_one();
    
        return result;
    }
    // auto enqueue(F task) -> std::future<decltype(task())>; //decltype 让编译器推导某个表达式的类型
    // auto ... -> 返回类型
    // 尾置返回类型（trailing return type）
    // 把函数返回类型写到参数列表后面。作用之一就是方便写这种依赖参数表达式的返回类型。
    // == future<decltype(task())> enqueue(F task)


private:

    // 工作线程
    std::vector<std::thread> workers;
 

    // 任务队列
    std::queue<std::function<void()>> tasks;


    // 保护任务队列
    std::mutex queue_mutex;


    // 通知worker
    std::condition_variable condition;


    // 是否停止
    bool stop;

};


#endif