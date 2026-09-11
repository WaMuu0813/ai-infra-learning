#include <iostream>
#include <unordered_map>
#include <string>

using namespace std;

int main()
{
    unordered_map<int, string> requests;

    requests[1] = "request_1";
    requests[2] = "request_2";

    auto it = requests.find(1);

    string* p = &it->second;

    cout << "before rehash:" << endl;
    cout << "bucket_count = " << requests.bucket_count() << endl;
    cout << "value address = " << p << endl;
    cout << "value = " << *p << endl;

    requests.rehash(100);

    cout << "\nafter rehash:" << endl;
    cout << "bucket_count = " << requests.bucket_count() << endl;
    cout << "value address = " << &requests.at(1) << endl;
    cout << "old p = " << p << endl;
    cout << "value = " << *p << endl;
    
    // cout << "initial bucket_count = "
    //      << requests.bucket_count() << endl;

    // for (int i = 0; i < 20; i++)
    // {
    //     requests[i] = "request_" + to_string(i);

    //     cout << "insert " << i
    //          << " | size = " << requests.size()
    //          << " | bucket_count = " << requests.bucket_count()
    //          << " | load_factor = " << requests.load_factor()
    //          << endl;
    // }

    return 0;
}