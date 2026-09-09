#include <iostream>
#include <utility>

using namespace std;

class TensorBuffer{
private:
    float* data;
    int size;

public:
    TensorBuffer(int n) {
        size = n;
        data = new float[size];

        cout << "Construct buffer, data = "
             << static_cast<void*>(data)
             << endl;
    }

    TensorBuffer(const TensorBuffer& other) {
        size = other.size;

        data = new float[size];

        for (int i = 0; i < size; i++) {
            data[i] = other.data[i];
        }

        cout << "Copy construct, new data = "
             << static_cast<void*>(data)
             << endl;
    }

    TensorBuffer(TensorBuffer&& other){
        size = other.size;
        data = other.data;

        other.size = 0;
        other.data = nullptr;

        cout << "Move construct, data = "
             << static_cast<void*>(data)
             << endl;
    }

    ~TensorBuffer() {
        cout << "Destruct buffer, data = "
             << static_cast<void*>(data)
             << endl;

        delete[] data;
    }

    void set(int index, float value) {
        data[index] = value;
    }

    void show(int index) {
        cout << data[index] << endl;
    }
};

int main(){
    TensorBuffer a(4);

    a.set(0, 3.14f);

    // TensorBuffer b = a;
    TensorBuffer b = move(a); //把表达式转换成一个 xvalue（将亡值），从而能够匹配右值引用

    // a.show(0);
    b.show(0);

    // TensorBuffer buffer(1024);

    // cout << "doing work..." << endl;

    return 0; 
}
