#include <iostream>
#include <vector>
#include <memory>


class Operator
{
public:
    virtual void execute() = 0;

    virtual ~Operator()
    {
        std::cout << "Operator destroyed\n";
    }
};


class Matmul : public Operator
{
public:
    void execute() override
    {
        std::cout << "Running Matmul\n";
    }


    ~Matmul()
    {
        std::cout << "Matmul destroyed\n";
    }
};


class Attention : public Operator
{
public:
    void execute() override
    {
        std::cout << "Running Attention\n";
    }


    ~Attention()
    {
        std::cout << "Attention destroyed\n";
    }
};


int main()
{
    std::vector<std::unique_ptr<Operator>> graph;


    graph.push_back(
        std::make_unique<Matmul>()
    );


    graph.push_back(
        std::make_unique<Attention>()
    );


    for(auto& op : graph)
    {
        op->execute();
    }


    return 0;
}