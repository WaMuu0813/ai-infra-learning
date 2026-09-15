#include <condition_variable>
#include <iostream>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>

struct Request {
    int id;

    explicit Request(int id) : id(id) {}
};

class BlockingQueue {
private:
    std::queue<std::shared_ptr<Request>> queue_;
    std::mutex mutex_;
    std::condition_variable cv_;

public:
    void push(std::shared_ptr<Request> request) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            queue_.push(std::move(request));
        }

        cv_.notify_one();
    }
    std::shared_ptr<Request> pop() {
        std::unique_lock<std::mutex> lock(mutex_);

        cv_.wait(lock, [this] {
            return !queue_.empty();
        });

        auto request = queue_.front();
        queue_.pop();

        return request;
    }
};

void worker(BlockingQueue& queue, int worker_id) {
    // auto request = queue.pop();

    // std::cout << "worker " << worker_id
    //           << " processing request "
    //           << request->id << '\n';
    while (true) {
        auto request = queue.pop();

        if (request->id == -1) {
            std::cout << "worker " << worker_id << " exiting\n";
            break;
        }

        std::cout << "worker " << worker_id
                  << " processing request "
                  << request->id << '\n';
    }
}

int main() {
    BlockingQueue queue;

    std::thread t1(worker, std::ref(queue), 1);
    std::thread t2(worker, std::ref(queue), 2);

    queue.push(std::make_shared<Request>(100));
    queue.push(std::make_shared<Request>(200));
    queue.push(std::make_shared<Request>(300));
    queue.push(std::make_shared<Request>(400));

    // shutdown signals
    queue.push(std::make_shared<Request>(-1));
    queue.push(std::make_shared<Request>(-1));

    t1.join();
    t2.join();

    std::cout << "all workers exited\n";

    return 0;
}