#define PASTER(x,y,z) x ## _ ## y ## _ ## z
#define EVALUATOR(x, y, z) PASTER(x, y, z)
#define NAME(fun, t1, t2) EVALUATOR(fun, t1, t2)