#include <iostream>
#include <memory>

using namespace std;


class Tensor {

public:

    Tensor(string n){
        name = n;
        cout << "Construct: " << name << endl;
    }


    ~Tensor(){
        cout << "Destruct: " << name << endl;
    }


    void show(){
        cout << "Tensor: " << name << endl;
    }


private:

    string name;
};



int main(){

    cout << "enter main" << endl;


    Tensor* raw = new Tensor("A");


    shared_ptr<Tensor> p1(raw);


    cout << "p1 count: "
         << p1.use_count()
         << endl;



    shared_ptr<Tensor> p2(raw);


    cout << "p1 count: "
         << p1.use_count()
         << endl;


    cout << "p2 count: "
         << p2.use_count()
         << endl;

    // shared_ptr<Tensor> p1 =
    //     make_shared<Tensor>("A");


    // cout << "count: "
    //      << p1.use_count()
    //      << endl;


    // Tensor* raw = p1.get();


    // cout << "raw pointer: "
    //      << raw
    //      << endl;


    // raw->show();


    // cout << "before delete shared_ptr"
    //      << endl;

    // cout << "enter main" << endl;


    // shared_ptr<Tensor> p1 = make_shared<Tensor>("A");


    // cout << "count: "
    //      << p1.use_count()
    //      << endl;


    // {
    //     shared_ptr<Tensor> p2 = p1;


    //     cout << "after copy count: "
    //          << p1.use_count()
    //          << endl;


    //     p2->show();
    // }


    // cout << "after scope count: "
    //      << p1.use_count()
    //      << endl;

    // delete raw;
    return 0;
}