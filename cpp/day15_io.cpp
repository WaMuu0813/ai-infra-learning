#include <fcntl.h>
#include <unistd.h>
#include <cerrno>
#include <iostream>

bool write_all(int fd, const char* buf, size_t total) {
    size_t written = 0;

    while (written < total) {
        ssize_t n = write(
            fd,
            buf + written,
            total - written
        );

        if (n > 0) {
            written += static_cast<size_t>(n);
        } else if (n == -1 && errno == EINTR) {
            continue;
        } else {
            return false;
        }
    }

    return true;
}

int main() {
    // const char* path = "/tmp/day15_io.txt";

    // int fd = open(path, O_RDONLY);

    // if (fd == -1) {
    //     std::cerr << "open failed\n";
    //     return 1;
    // }

    // char buf[4];

    // while (true) {
    //     ssize_t n = read(fd, buf, sizeof(buf));

    //     if (n > 0) {
    //         std::cout << "read " << n << " bytes: "; 
    //         std::cout.write(buf, n);
    //         std::cout << '\n';
    //     } else if (n == 0) {
    //         std::cout << "EOF\n";
    //         break;
    //     } else {
    //         std::cerr << "read failed\n";
    //         break;
    //     }
    // }

    // // char buf[8] = {};

    // // ssize_t n1 = read(fd, buf, 7);

    // // std::cout << "first read:\n";
    // // std::cout << "n1 = " << n1 << '\n';
    // // std::cout << "buf = " << buf << '\n';

    // // ssize_t n2 = read(fd, buf, 7);

    // // std::cout << "\nsecond read:\n";
    // // std::cout << "n2 = " << n2 << '\n';
    // // std::cout << "buf = " << buf << '\n';

    // close(fd);
    // return 0;
    const char* path = "/tmp/day15_output.txt";

    int fd = open(
        path,
        O_WRONLY | O_CREAT | O_TRUNC,
        0644
    );

    if (fd == -1) {
        std::cerr << "open failed\n";
        return 1;
    }

    const char data[] = "Hello AI Infra!\n";

    bool ok = write_all(
        fd,
        data,
        sizeof(data) - 1
    );

    close(fd);

    std::cout << "write_all: "
              << (ok ? "success" : "failed")
              << '\n';

    return ok ? 0 : 1;
}

