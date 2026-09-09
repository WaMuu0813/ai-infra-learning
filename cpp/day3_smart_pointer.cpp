#include <iostream>
#include <memory>
#include <string>

using namespace std;

class Tensor {
public:
    string name;

    Tensor(string n) {
        name = n;
        cout << "Construct: " << name << endl;
    }

    ~Tensor() {
        cout << "Destruct: " << name << endl;
    }

    void show() {
        cout << "Tensor: " << name << endl;
    }
};

int main() {

    auto p1 = make_unique<Tensor>("A");

    cout << "before move" << endl;
    p1->show();

    // auto p2 = p1;
    auto p2 = std::move(p1);

    cout << "after move" << endl;

    if (p1 == nullptr) {
        cout << "p1 is nullptr" << endl;
    }

    p2->show();

    // cout << "enter main" << endl;

    // unique_ptr<Tensor> p = make_unique<Tensor>("A");

    // p->show();

    // cout << "leave main" << endl;

    return 0;
}