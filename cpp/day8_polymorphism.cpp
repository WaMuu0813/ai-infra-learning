#include <iostream>
#include <memory>

class Backend {
public:
    virtual void run() = 0;

    virtual ~Backend() {
        std::cout << "~Backend()\n";
    }
};

class CPUBackend : public Backend {
public:
    void run() override {
        std::cout << "CPUBackend::run()\n";
    }

    ~CPUBackend() override {
        std::cout << "~CPUBackend()\n";
    }
};

int main() {
    std::unique_ptr<Backend> backend =
        std::make_unique<CPUBackend>();

    backend->run();

    return 0;
}