#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <iostream>

int main() {
    // 忽略 SIGPIPE，不让它直接终止进程
    signal(SIGPIPE, SIG_IGN);

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
        close(pipefd[0]);
        close(pipefd[1]);
        return 0;
    } else {
        close(pipefd[0]);

        // 确保子进程已经退出，所有 reader 都消失
        wait(nullptr);

        std::cout << "before write\n";

        ssize_t n = write(pipefd[1], "hello", 5);

        std::cout << "after write\n";
        std::cout << "n = " << n << '\n';

        if (n == -1) {
            std::cout << "errno = " << errno << '\n';
            std::cout << "error = " << std::strerror(errno) << '\n';
        }

        close(pipefd[1]);
    }

    return 0;
}

// int main() {
//     int pipefd[2];

//     if (pipe(pipefd) == -1) {
//         std::cerr << "pipe failed\n";
//         return 1;
//     }

//     pid_t pid = fork();

//     if (pid == -1) {
//         std::cerr << "fork failed\n";
//         return 1;
//     }

//     if (pid == 0) {
//         // 子进程：关闭自己持有的两个端
//         close(pipefd[0]);
//         close(pipefd[1]);

//         return 0;
//     } else {
//         // 父进程不读
//         close(pipefd[0]);

//         // 确保子进程已经退出并关闭它继承的两个 fd
//         wait(nullptr);

//         std::cout << "before write\n";

//         ssize_t n = write(pipefd[1], "hello", 5);

//         std::cout << "after write, n = " << n << '\n';

//         close(pipefd[1]);
//     }

//     return 0;
// }
