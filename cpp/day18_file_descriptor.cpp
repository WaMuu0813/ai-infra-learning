#include <fcntl.h>
#include <iostream>
#include <unistd.h>
#include <utility>

class FileDescriptor {
  private:
    int fd_;

  public:
    explicit FileDescriptor(int fd) : fd_(fd) {}

    FileDescriptor(FileDescriptor &&other) noexcept : fd_(other.fd_) {
        other.fd_ = -1;
    }

    FileDescriptor &operator=(FileDescriptor &&other) noexcept {
        if (this != &other) {
            if (fd_ >= 0) {
                ::close(fd_);
            }

            fd_ = other.fd_;
            other.fd_ = -1;
        }

        return *this;
    }

    ~FileDescriptor() {
        if (fd_ >= 0) {
            std::cout << "closing fd " << fd_ << '\n';
            ::close(fd_);
        }
    }

    int get() const { return fd_; }
};

int main() {
    int raw_fd1 =
        ::open("/tmp/day18_a.txt", O_CREAT | O_WRONLY | O_TRUNC, 0644);

    int raw_fd2 =
        ::open("/tmp/day18_b.txt", O_CREAT | O_WRONLY | O_TRUNC, 0644);

    if (raw_fd1 < 0 || raw_fd2 < 0) {
        std::cerr << "open failed\n";

        if (raw_fd1 >= 0) {
            ::close(raw_fd1);
        }

        if (raw_fd2 >= 0) {
            ::close(raw_fd2);
        }

        return 1;
    }

    FileDescriptor a(raw_fd1);
    FileDescriptor b(raw_fd2);

    std::cout << "before move assignment\n";
    std::cout << "a fd = " << a.get() << '\n';
    std::cout << "b fd = " << b.get() << '\n';

    b = std::move(a);

    std::cout << "after move assignment\n";
    std::cout << "a fd = " << a.get() << '\n';
    std::cout << "b fd = " << b.get() << '\n';

    return 0;
}

// int main() {
//     FileDescriptor a(3);

//     std::cout << "a fd before move: " << a.get() << '\n';

//     FileDescriptor b = std::move(a);

//     std::cout << "a fd after move: " << a.get() << '\n';

//     std::cout << "b fd after move: " << b.get() << '\n';

//     return 0;
// }