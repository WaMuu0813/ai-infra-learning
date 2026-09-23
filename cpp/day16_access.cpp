// #include <iostream>

// class Base {
// public:
//     int public_value = 10;

// protected:
//     int protected_value = 20;

// private:
//     int private_value = 30;
// };

// class Derived : public Base {
// public:
//     void test_inside() {
//         std::cout << public_value << '\n';
//         std::cout << protected_value << '\n';

//         // // 先不要取消注释
//         // std::cout << private_value << '\n';
//     }
// };

// int main() {
//     Derived d;

//     std::cout << d.public_value << '\n';

//     // // 先不要取消注释
//     // std::cout << d.protected_value << '\n';

//     d.test_inside();
// }


#include <iostream>

class Base {
public:
    int a = 10;

protected:
    int b = 20;
};

class PublicDerived : public Base {
public:
    void test() {
        std::cout << "PublicDerived: "
                  << a << " " << b << '\n';
    }
};

class ProtectedDerived : protected Base {
public:
    void test() {
        std::cout << "ProtectedDerived: "
                  << a << " " << b << '\n';
    }
};

class PrivateDerived : private Base {
public:
    void test() {
        std::cout << "PrivateDerived: "
                  << a << " " << b << '\n';
    }
};

int main() {
    PublicDerived d1;
    ProtectedDerived d2;
    PrivateDerived d3;

    d1.test();
    d2.test();
    d3.test();

    std::cout << d1.a << '\n';

    // std::cout << d2.a << '\n';
    // std::cout << d3.a << '\n';

    return 0;
}