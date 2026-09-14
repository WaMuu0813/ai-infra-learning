class Backend:
    def __init__(self, name):
        self.name = name

    def run(self):
        print("Backend::run")


class CPUBackend(Backend):
    def __init__(self, name, threads):
        super().__init__(name)
        self.threads = threads

    def run(self):
        print(f"{self.name}: CPUBackend::run, threads={self.threads}")


backend = CPUBackend("cpu_backend", 8)

print(backend.name)
print(backend.threads)
backend.run()