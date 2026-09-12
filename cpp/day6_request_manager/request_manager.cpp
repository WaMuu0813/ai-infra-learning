#include "request_manager.h"
#include <iostream>

Request::Request(int id) : id_(id) {
}

int Request::get_id() const {
    return id_;
}

void RequestManager::add_request(
    const std::shared_ptr<Request>& request
) {

    std::cout << "Adding request: " << request->get_id() << '\n';

    queue_.push_back(request);
    request_table_[request->get_id()] = request;
}

std::shared_ptr<Request> RequestManager::find_request(int id) {
    auto it = request_table_.find(id);

    if (it == request_table_.end()) {
        return nullptr;
    }

    return it->second.lock();
}