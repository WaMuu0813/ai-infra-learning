#include <iostream>
#include <memory>
#include <vector>

class Request {
public:
    explicit Request(int id) : id_(id) {
        std::cout << "Construct " << id_ << '\n';
    }

    ~Request() {
        std::cout << "Destruct " << id_ << '\n';
    }

    int get_id() const {
        return id_;
    }

private:
    int id_;
};

int main() {
    std::vector<std::shared_ptr<Request>> queue;

    auto p = std::make_shared<Request>(101);

    Request* raw = p.get();

    std::shared_ptr<Request> p2(raw);

    queue.push_back(p);

    std::cout << "p count = "
              << p.use_count()
              << '\n';

    std::cout << "p2 count = "
              << p2.use_count()
              << '\n';

    return 0;
}