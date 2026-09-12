#pragma once

#include <memory>
#include <unordered_map>
#include <vector>


class Request {
public:
    explicit Request(int id);

    int get_id() const;

private:
    int id_;
};


class RequestManager {
public:
    void add_request(const std::shared_ptr<Request>& request);

    std::shared_ptr<Request> find_request(int id);

private:
    std::vector<std::shared_ptr<Request>> queue_;
    std::unordered_map<int, std::weak_ptr<Request>> request_table_;
};