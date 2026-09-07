#include <iostream>

int main()
{
    int a = 10;


    int* p = &a;


    std::cout << "before: "
              << a
              << std::endl;


    *p = 100;


    std::cout << "after: "
              << a
              << std::endl;


    return 0;
}
