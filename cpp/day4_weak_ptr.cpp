#include <iostream>
#include <memory>

using namespace std;


class Tensor{

public:

    Tensor(string n)
    {
        name=n;
        cout<<"Construct: "
            <<name<<endl;
    }


    ~Tensor()
    {
        cout<<"Destruct: "
            <<name<<endl;
    }


    void show()
    {
        cout<<"Tensor: "
            <<name<<endl;
    }


private:

    string name;

};



int main(){

    shared_ptr<Tensor> p =
        make_shared<Tensor>("A");


    cout<<"shared count: "
        <<p.use_count()
        <<endl;



    weak_ptr<Tensor> w=p;


    cout<<"after weak:"
        <<p.use_count()
        <<endl;


    return 0;
}