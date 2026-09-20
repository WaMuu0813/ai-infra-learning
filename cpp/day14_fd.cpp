#include <fcntl.h>
#include <unistd.h>
#include <iostream>

void print_offset(int fd, const char* name) {
    off_t pos = lseek(fd, 0, SEEK_CUR);
    std::cout << name << " offset = " << pos << '\n';
}

int main() {
    const char* path = "/tmp/fd_demo.txt";

    int fd1 = open(path, O_RDONLY);
    if (fd1 == -1) {
        std::cerr << "open failed\n";
        return 1;
    }

    int fd2 = dup(fd1);
    // int fd2 = open(path, O_RDONLY);
    if (fd2 == -1) {
        std::cerr << "dup failed\n";
        close(fd1);
        return 1;
    }

    std::cout << "fd1 = " << fd1 << '\n';
    std::cout << "fd2 = " << fd2 << '\n';

    print_offset(fd1, "fd1");
    print_offset(fd2, "fd2");

    char buf[6] = {};

    read(fd1, buf, 5);

    std::cout << "\nafter read(fd1, 5):\n";
    std::cout << "buf = " << buf << '\n';

    print_offset(fd1, "fd1");
    print_offset(fd2, "fd2");

    std::cout << "\nPID = " << getpid() << '\n';
    std::cout << "Press Enter to exit...";
    std::cin.get();

    close(fd2);
    close(fd1);



    return 0;
}