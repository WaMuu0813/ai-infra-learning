#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <iostream>

int main() {
    const char* path = "/tmp/mmap_demo.txt";

    // int fd = open(path, O_RDONLY);
    int fd = open(path, O_RDWR);

    if (fd == -1) {
        std::cerr << "open failed\n";
        return 1;
    }

    struct stat st {};

    if (fstat(fd, &st) == -1) {
        std::cerr << "fstat failed\n";
        close(fd);
        return 1;
    }

    std::size_t size = static_cast<std::size_t>(st.st_size);

    void* addr = mmap(
        nullptr,
        size,
        // PROT_READ,
        PROT_READ | PROT_WRITE,
        // MAP_PRIVATE,
        MAP_SHARED,
        fd,
        0
    );

    if (addr == MAP_FAILED) {
        std::cerr << "mmap failed\n";
        close(fd);
        return 1;
    }

    // std::cout << "mapped address = " << addr << '\n';

    // const char* data = static_cast<const char*>(addr);

    // std::cout << "content = ";
    // std::cout.write(data, size);

    // munmap(addr, size);

    std::cout << "PID = " << getpid() << '\n';
    std::cout << "mapped address = " << addr << '\n';

    // const char* data = static_cast<const char*>(addr);
    char* data = static_cast<char*>(addr);

    std::cout << "content = ";
    // std::cout.write(data, size);
    std::cout << "before write = ";
    std::cout.write(data, size);

    data[0] = 'H';

    if (msync(addr, size, MS_SYNC) == -1) {
        std::cerr << "msync failed\n";
    }

    std::cout << "after write  = ";
    std::cout.write(data, size);

    std::cout << "\nPress Enter to exit...";
    std::cin.get();

    munmap(addr, size);
    close(fd);

    return 0;
}