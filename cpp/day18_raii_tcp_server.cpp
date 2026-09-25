#include <arpa/inet.h>
#include <cerrno>
#include <iostream>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#include <utility>

class FileDescriptor {
  private:
    int fd_;

  public:
    explicit FileDescriptor(int fd = -1) : fd_(fd) {}

    ~FileDescriptor() {
        if (fd_ >= 0) {
            std::cout << "closing fd " << fd_ << '\n';
            ::close(fd_);
        }
    }

    FileDescriptor(const FileDescriptor &) = delete;
    FileDescriptor &operator=(const FileDescriptor &) = delete;

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

    int get() const { return fd_; }

    bool valid() const { return fd_ >= 0; }
};

ssize_t read_exact(int fd, void *buffer, size_t count) {
    char *ptr = static_cast<char *>(buffer);
    size_t total = 0;

    while (total < count) {
        ssize_t n = ::read(fd, ptr + total, count - total);

        if (n > 0) {
            total += static_cast<size_t>(n);
        } else if (n == 0) {
            break;
        } else {
            if (errno == EINTR) {
                continue;
            }
            return -1;
        }
    }

    return static_cast<ssize_t>(total);
}

int main() {
    // 1. 创建监听 socket
    FileDescriptor listen_fd(::socket(AF_INET, SOCK_STREAM, 0));

    if (!listen_fd.valid()) {
        std::perror("socket");
        return 1;
    }

    // 2. 准备服务器地址
    sockaddr_in server_addr{};
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(8080);
    server_addr.sin_addr.s_addr = htonl(INADDR_ANY);

    // 3. 把监听 socket 绑定到 0.0.0.0:8080
    if (::bind(listen_fd.get(), reinterpret_cast<sockaddr *>(&server_addr),
               sizeof(server_addr)) < 0) {

        std::perror("bind");
        return 1;
    }

    // 4. 进入监听状态
    if (::listen(listen_fd.get(), 16) < 0) {
        std::perror("listen");
        return 1;
    }

    std::cout << "server listening on 0.0.0.0:8080\n";

    // 5. 准备接收客户端地址
    sockaddr_in client_addr{};
    socklen_t client_addr_len = sizeof(client_addr);

    // 6. accept 一个客户端
    FileDescriptor conn_fd(::accept(listen_fd.get(),
                                    reinterpret_cast<sockaddr *>(&client_addr),
                                    &client_addr_len));

    if (!conn_fd.valid()) {
        std::perror("accept");
        return 1;
    }

    // 7. 把客户端二进制 IP 转成人能看的字符串
    char client_ip[INET_ADDRSTRLEN]{};

    ::inet_ntop(AF_INET, &client_addr.sin_addr, client_ip, sizeof(client_ip));

    std::cout << "client connected: " << client_ip << ":"
              << ntohs(client_addr.sin_port) << '\n';

    // 8. 从客户端读取数据
    char buffer[1024]{};

    while (true) {
        ssize_t n = ::read(conn_fd.get(), buffer, sizeof(buffer));

        if (n > 0) {
            std::cout << "received " << n << " bytes: ";

            std::cout.write(buffer, n);
            std::cout << '\n';
        } else if (n == 0) {
            std::cout << "client closed connection\n";
            break;
        } else {
            std::perror("read");
            return 1;
        }
    }

    return 0;
}