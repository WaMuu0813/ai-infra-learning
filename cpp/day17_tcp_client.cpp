#include <arpa/inet.h>
#include <iostream>
#include <sys/socket.h>
#include <unistd.h>

int main() {
    int sock_fd = socket(AF_INET, SOCK_STREAM, 0);

    if (sock_fd == -1) {
        std::cerr << "socket failed\n";
        return 1;
    }

    sockaddr_in server_addr{};
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(8080);

    if (inet_pton(AF_INET, "127.0.0.1", &server_addr.sin_addr) != 1) {
        std::cerr << "invalid address\n";
        close(sock_fd);
        return 1;
    }

    if (connect(sock_fd, reinterpret_cast<sockaddr *>(&server_addr),
                sizeof(server_addr)) == -1) {
        std::cerr << "connect failed\n";
        close(sock_fd);
        return 1;
    }

    std::cout << "connected to server\n";

    const char *message = "hello from client";

    ssize_t n = write(sock_fd, message, 17);

    std::cout << "client wrote " << n << " bytes\n";

    close(sock_fd);

    std::cin.get();

    close(sock_fd);
    return 0;
}