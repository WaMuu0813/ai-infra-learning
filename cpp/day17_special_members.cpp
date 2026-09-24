#include <iostream>

// class Resource {
//   public:
//     Resource() { std::cout << "constructor\n"; }

//     Resource(const Resource &) = delete;

//     ~Resource() { std::cout << "destructor\n"; }
// };

class Resource {
  public:
    Resource() = default;

    Resource(const Resource &) = delete;

    ~Resource() = default;
};

int main() {
    Resource a;

    Resource b = a;

    return 0;
}