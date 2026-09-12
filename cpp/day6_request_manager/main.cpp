#include "request_manager.h"

#include <iostream>
#include <memory>


int main() {
    RequestManager manager;

    auto request1 = std::make_shared<Request>(101);
    auto request2 = std::make_shared<Request>(102);

    manager.add_request(request1);
    manager.add_request(request2);

    auto found = manager.find_request(101);

    if (found) {
        std::cout << "Found request: "
                  << found->get_id()
                  << '\n';
    }

    return 0;
}