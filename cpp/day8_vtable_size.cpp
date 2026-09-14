#include <iostream>

class Empty {
};

class Normal {
public:
    void run() {
    }
};

class Virtual {
public:
    virtual void run() {
    }
};

class ManyVirtual {
public:
    virtual void f1() {}
    virtual void f2() {}
    virtual void f3() {}
    virtual void f4() {}
    virtual void f5() {}
};

int main() {
    std::cout << "sizeof(Empty)   = "
              << sizeof(Empty) << '\n';

    std::cout << "sizeof(Normal)  = "
              << sizeof(Normal) << '\n';

    std::cout << "sizeof(Virtual) = "
              << sizeof(Virtual) << '\n';

    std::cout << "sizeof(ManyVirtual) = "
              << sizeof(ManyVirtual) << '\n';

    return 0;
}