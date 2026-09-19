#include <iostream>
#include <unistd.h>

int global_value = 100;

void foo() {
    std::cout << "foo is running\n";
}

int main() {
    int local_value = 200;
    int* heap_value = new int(300);

    std::cout << "PID        = " << getpid() << '\n';
    std::cout << "function   = "
              << reinterpret_cast<void*>(&foo) << '\n';
    std::cout << "global     = " << &global_value << '\n';
    std::cout << "heap       = " << heap_value << '\n';
    std::cout << "&heap ptr  = " << &heap_value << '\n';
    std::cout << "stack      = " << &local_value << '\n';

    std::cout << "\nPress Enter to exit...";
    std::cin.get();

    delete heap_value;
    return 0;
}