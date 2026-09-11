#include <iostream>
#include <vector>

using namespace std;

class Tensor
{
public:
    string name;

    Tensor(string n) : name(n)
    {
        cout << "Construct: " << name << endl;
    }

    Tensor(const Tensor& other) : name(other.name)
    {
        cout << "Copy: " << name << endl;
    }

    // Tensor(Tensor&& other) noexcept : name(std::move(other.name))
    Tensor(Tensor&& other) : name(std::move(other.name))
    {
        cout << "Move: " << name << endl;
    }

    ~Tensor()
    {
        cout << "Destruct: " << name << endl;
    }
};

int main()
{
    // vector<int> nums;

    // cout << "size = " << nums.size()
    //      << ", capacity = " << nums.capacity()
    //      << endl;

    // for (int i = 0; i < 20; i++)
    // {
    //     nums.push_back(i);

    //     cout << "push " << i
    //          << " | size = " << nums.size()
    //          << " | capacity = " << nums.capacity()
    //          << " | data = " << nums.data()
    //          << endl;
    // }

    // nums.reserve(4);

    // nums.push_back(10);
    // nums.push_back(20);
    // nums.push_back(30);
    // nums.push_back(40);

    // int* p = &nums[0];

    // cout << "before:" << endl;
    // cout << "capacity = " << nums.capacity() << endl;
    // cout << "data = " << nums.data() << endl;
    // cout << "p = " << p << endl;

    // nums.push_back(50);

    // cout << "\nafter:" << endl;
    // cout << "capacity = " << nums.capacity() << endl;
    // cout << "data = " << nums.data() << endl;
    // cout << "p = " << p << endl;

    vector<Tensor> tensors;

    tensors.reserve(2);

    cout << "---- push A ----" << endl;
    tensors.emplace_back("A");

    cout << "---- push B ----" << endl;
    tensors.emplace_back("B");

    cout << "---- push C ----" << endl;
    tensors.emplace_back("C");

    cout << "---- end ----" << endl;

    return 0;
}