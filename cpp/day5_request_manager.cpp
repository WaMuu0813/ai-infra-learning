#include <iostream>
#include <vector>
#include <unordered_map>
#include <memory>
#include <string>

using namespace std;

class Request
{
public:
    int id;
    string prompt;

    Request(int id, string prompt)
        : id(id), prompt(prompt)
    {
        cout << "Construct Request " << id << endl;
    }

    ~Request()
    {
        cout << "Destruct Request " << id << endl;
    }
};

int main()
{
    vector<shared_ptr<Request>> queue;

    // unordered_map<int, shared_ptr<Request>> request_table;
    unordered_map<int, weak_ptr<Request>> request_table;

    auto r1 = make_shared<Request>(101, "hello");
    auto r2 = make_shared<Request>(666, "AI Infra");

    queue.push_back(r1);
    queue.push_back(r2);

    request_table[r1->id] = r1;
    request_table[r2->id] = r2;

    cout << "r1 use_count = " << r1.use_count() << endl;
    cout << "r2 use_count = " << r2.use_count() << endl;

    queue.erase(queue.begin());
    r1.reset();

    auto request = request_table[101].lock();

    if (request)
    {
        cout << "Request still alive" << endl;
    }
    else
    {
        cout << "Request expired" << endl;
    }

    // auto it = request_table.find(101);

    // if (it != request_table.end())
    // {
    //     auto request = it->second.lock();

    //     if (request)
    //     {
    //         cout << "Found request: "
    //              << request->id << " "
    //              << request->prompt << endl;
    //     }
    //     else
    //     {
    //         cout << "Request expired" << endl;
    //     }
    // }

    // cout << "\n---- erase from table ----" << endl;

    // request_table.erase(101);

    // cout << "r1 use_count = " << r1.use_count() << endl;

    // cout << "\n---- erase from queue ----" << endl;

    // queue.erase(queue.begin());

    // cout << "r1 use_count = " << r1.use_count() << endl;

    // cout << "\n---- reset r1 ----" << endl;

    // r1.reset();

    cout << "end main" << endl;
    

    return 0;
}