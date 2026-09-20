// 一个类型的对象地址要满足这个类型的 alignment requirement（对齐要求），
// 由 alignof(T) 表示，而不一定等于 sizeof(T)

// sizeof(struct) 通常需要是该结构体自身 alignment 的整数倍；
// 而结构体自身的 alignment 通常由成员中最大的 alignment 决定。 