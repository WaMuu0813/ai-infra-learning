#include <iostream>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

int main() {
    int listen_fd = socket(AF_INET, SOCK_STREAM, 0);

    if (listen_fd == -1) {
        std::cerr << "socket failed\n";
        return 1;
    }

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    addr.sin_port = htons(8080);

    if (bind(listen_fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) ==
        -1) {
        std::cerr << "bind failed\n";
        close(listen_fd);
        return 1;
    }

    if (listen(listen_fd, 16) == -1) {
        std::cerr << "listen failed\n";
        close(listen_fd);
        return 1;
    }

    std::cout << "server listening on port 8080\n";

    int conn_fd = accept(listen_fd, nullptr, nullptr);

    if (conn_fd == -1) {
        std::cerr << "accept failed\n";
        close(listen_fd);
        return 1;
    }

    std::cout << "client connected\n";
    std::cout << "listen_fd = " << listen_fd << '\n';
    std::cout << "conn_fd   = " << conn_fd << '\n';

    char buffer[1024];

    ssize_t n = read(conn_fd, buffer, sizeof(buffer));

    if (n > 0) {
        std::cout << "server read " << n << " bytes: ";
        std::cout.write(buffer, n);
        std::cout << '\n';
    }

    close(conn_fd);
    close(listen_fd);
    return 0;
}