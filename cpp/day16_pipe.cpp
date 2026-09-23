#include <cstring>
#include <iostream>
#include <sys/wait.h>
#include <unistd.h>

// int main() {
//     int pipefd[2];

//     if (pipe(pipefd) == -1) {
//         std::cerr << "pipe failed\n";
//         return 1;
//     }

//     const char* msg = "hello pipe";

//     ssize_t written = write(
//         pipefd[1],
//         msg,
//         std::strlen(msg)
//     );

//     char buffer[64];

//     ssize_t n = read(
//         pipefd[0],
//         buffer,
//         sizeof(buffer)
//     );

//     std::cout << "written = " << written << '\n';
//     std::cout << "read    = " << n << '\n';

//     if (n > 0) {
//         std::cout.write(buffer, n);
//         std::cout << '\n';
//     }

//     close(pipefd[0]);
//     close(pipefd[1]);

//     return 0;
// }

int main() {
    int pipefd[2];

    if (pipe(pipefd) == -1) {
        std::cerr << "pipe failed\n";
        return 1;
    }

    pid_t pid = fork();

    if (pid == -1) {
        std::cerr << "fork failed\n";
        return 1;
    }

    if (pid == 0) {
        // 子进程：只负责读
        close(pipefd[1]);

        // char buffer[64];

        // ssize_t n = read(
        //     pipefd[0],
        //     buffer,
        //     sizeof(buffer)
        // );

        // if (n > 0) {
        //     std::cout << "child received: ";
        //     std::cout.write(buffer, n);
        //     std::cout << '\n';
        // }

        char buffer[8];

        while (true) {
            ssize_t n = read(pipefd[0], buffer, sizeof(buffer));

            if (n > 0) {
                std::cout << "child read " << n << " bytes: ";
                std::cout.write(buffer, n);
                std::cout << '\n';
            } else if (n == 0) {
                std::cout << "child: EOF\n";
                break;
            } else {
                std::cerr << "read failed\n";
                break;
            }
        }
        close(pipefd[0]);
    } else {
        // 父进程：只负责写
        close(pipefd[0]);

        const char *msg = "hello from parent";

        write(pipefd[1], msg, std::strlen(msg));

        close(pipefd[1]);

        wait(nullptr);
    }

    return 0;
}