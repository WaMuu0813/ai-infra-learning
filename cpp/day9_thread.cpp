// #include <iostream>
// #include <thread>

// // void worker(int id) {
// //     std::cout << "worker " << id << " is running\n";
// // }

// // int main() {
// //     std::cout << "main start\n";

// //     std::thread t1(worker, 1);
// //     std::thread t2(worker, 2);

// //     t1.join();
// //     t2.join();

// //     std::cout << "main end\n";

// //     return 0;
// // }   

// int counter = 0;

// void worker() {
//     for (int i = 0; i < 1000000; ++i) {
//         counter++;
//     }
// }

// int main() {
//     std::thread t1(worker);
//     std::thread t2(worker);

//     t1.join();
//     t2.join();

//     std::cout << "counter = " << counter << '\n';

//     return 0;
// }

#include <iostream>
#include <thread>
#include <mutex>

int counter = 0;
std::mutex counter_mutex;

void worker() {
    // std::lock_guard<std::mutex> lock(counter_mutex);

    // for (int i = 0; i < 1000000; ++i) {
    //     counter++;
    // }
    for (int i = 0; i < 1000000; ++i) {
        std::lock_guard<std::mutex> lock(counter_mutex);
        counter++;
    }
}

int main() {
    std::thread t1(worker);
    std::thread t2(worker);

    t1.join();
    t2.join();

    std::cout << "counter = " << counter << '\n';

    return 0;
}