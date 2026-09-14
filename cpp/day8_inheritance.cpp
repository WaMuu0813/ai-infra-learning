#include <iostream>
#include <string>
#include <memory>

class Backend {
public:
    // virtual void run() = 0{
    //     std::cout << "Backend::run()" << '\n';
    // }
    virtual void run() = 0;

    virtual ~Backend() {
        std::cout << "~Backend()" << '\n';
    }
};

class CPUBackend : public Backend {
public:
    void run() override {
        std::cout << "CPUBackend::run()" << '\n';
    }

     ~CPUBackend() {
        std::cout << "~CPUBackend()" << '\n';
    }
};

int main() {
    // CPUBackend cpu;

    // cpu.run();

    // Backend* p = &cpu;

    // Backend* p = new CPUBackend();
    std::unique_ptr<Backend> p = std::make_unique<CPUBackend>();
    p->run();

    return 0;
}