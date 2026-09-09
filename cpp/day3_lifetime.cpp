#include <iostream>
#include <string>

using namespace std;

class Tensor{
public:
    string name;

    Tensor(string n){
        name = n;
        cout << "Construct: " << name << endl;
    }

    ~Tensor(){
        cout << "Destruct: " << name << endl;
    }
};

void test(){
    cout << "enter test()" << endl;

    Tensor a("A");

    cout << "leave test()" << endl;
}

int main(){
    cout << "enter main()" << endl;

    Tensor a("stack-A");

    Tensor* b = new Tensor("heap-B");

    cout << "leave main()" << endl;

    delete b;

    cout << b << endl;

    return 0;
}