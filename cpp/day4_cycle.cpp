#include <iostream>
#include <memory>

using namespace std;


class Node {

public:

    string name;

    // shared_ptr<Node> next;
    weak_ptr<Node> next;


    Node(string n)
    {
        name = n;
        cout << "Construct: "
             << name
             << endl;
    }


    ~Node()
    {
        cout << "Destruct: "
             << name
             << endl;
    }

};



int main(){

    {
        shared_ptr<Node> a =
            make_shared<Node>("A");


        shared_ptr<Node> b =
            make_shared<Node>("B");


        a->next = b;

        b->next = a;


        cout << "a count: "
             << a.use_count()
             << endl;


        cout << "b count: "
             << b.use_count()
             << endl;

    }


    cout << "end main"
         << endl;


    return 0;
}