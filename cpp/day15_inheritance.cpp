// #include <iostream>

// class CPU {
// public:
//     int cpu_id = 10;

//     void run_cpu() {
//         std::cout << "CPU running\n";
//     }
// };

// class GPU {
// public:
//     int gpu_id = 20;

//     void run_gpu() {
//         std::cout << "GPU running\n";
//     }
// };

// // class AIWorker : public CPU, public GPU {
// // };

// class Device {
// public:
//     int id = 100;
// };

// // class CPUDevice : public Device {
// // };

// // class GPUDevice : public Device {
// // };

// class CPUDevice : virtual public Device {
// };

// class GPUDevice : virtual public Device {
// };

// class AIWorker : public CPUDevice, public GPUDevice {
// };

// int main() {
//     AIWorker worker;

//     // std::cout << worker.cpu_id << '\n';
//     // std::cout << worker.gpu_id << '\n';

//     // worker.run_cpu();
//     // worker.run_gpu();

//     // std::cout << "AIWorker address: "
//     //           << static_cast<void*>(&worker) << '\n';

//     // std::cout << "CPU address:      "
//     //           << static_cast<void*>(
//     //                  static_cast<CPU*>(&worker)
//     //              ) << '\n';

//     // std::cout << "GPU address:      "
//     //           << static_cast<void*>(
//     //                  static_cast<GPU*>(&worker)
//     //              ) << '\n';


//     std::cout << "sizeof(Device)    = "
//               << sizeof(Device) << '\n';

//     std::cout << "sizeof(CPUDevice) = "
//               << sizeof(CPUDevice) << '\n';

//     std::cout << "sizeof(GPUDevice) = "
//               << sizeof(GPUDevice) << '\n';

//     std::cout << "sizeof(AIWorker)  = "
//               << sizeof(AIWorker) << '\n';

//     // std::cout << worker.id << '\n';
//     worker.CPUDevice::id = 111;
//     worker.GPUDevice::id = 222;
    
//     std::cout << "worker.id = "
//           << worker.id << '\n';

//     std::cout << "CPU path id = "
//               << worker.CPUDevice::id << '\n';
    
//     std::cout << "GPU path id = "
//               << worker.GPUDevice::id << '\n';

//     Device* p = &worker;

//     std::cout << "direct Device address = "
//               << static_cast<void*>(p) << '\n';
    
//     Device* device_from_cpu =
//         static_cast<CPUDevice*>(&worker);
    
//     Device* device_from_gpu =
//         static_cast<GPUDevice*>(&worker);
    
//     std::cout << "Device via CPU: "
//               << static_cast<void*>(device_from_cpu) << '\n';
    
//     std::cout << "Device via GPU: "
//               << static_cast<void*>(device_from_gpu) << '\n';
// }

#include <iostream>

class Device {
public:
    int id = 100;
    Device(int id) {
        std::cout << "Device(" << id << ")\n";
    }

    ~Device() {
        std::cout << "~Device()\n";
    }
};

class CPUDevice : virtual public Device {
public:
    CPUDevice()
        : Device(111) {
        std::cout << "CPUDevice()\n";
    }

    ~CPUDevice() {
        std::cout << "~CPUDevice()\n";
    }
};

class GPUDevice : virtual public Device {
public:
    GPUDevice()
        : Device(222) {
        std::cout << "GPUDevice()\n";
    }

    ~GPUDevice() {
        std::cout << "~GPUDevice()\n";
    }
};

class AIWorker : public CPUDevice, public GPUDevice {
public:
    AIWorker()
        : Device(999) {
        std::cout << "AIWorker()\n";
    }

    ~AIWorker() {
        std::cout << "~AIWorker()\n";
    }
};

int main() {
    AIWorker worker;
    
    std::cout << "\n--- addresses ---\n";

    std::cout << "AIWorker  = "
              << static_cast<void*>(&worker) << '\n';

    std::cout << "CPUDevice = "
              << static_cast<void*>(
                     static_cast<CPUDevice*>(&worker)
                 ) << '\n';

    std::cout << "GPUDevice = "
              << static_cast<void*>(
                     static_cast<GPUDevice*>(&worker)
                 ) << '\n';

    std::cout << "Device    = "
              << static_cast<void*>(
                     static_cast<Device*>(&worker)
                 ) << '\n';

    // std::cout << "\n--- sizeof ---\n";

    // std::cout << "sizeof(Device)    = "
    //           << sizeof(Device) << '\n';

    // std::cout << "sizeof(CPUDevice) = "
    //           << sizeof(CPUDevice) << '\n';

    // std::cout << "sizeof(GPUDevice) = "
    //           << sizeof(GPUDevice) << '\n';

    // std::cout << "sizeof(AIWorker)  = "
    //           << sizeof(AIWorker) << '\n';
}