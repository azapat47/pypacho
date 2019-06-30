#define PASTER(x,y,z) x ## _ ## y ## _ ## z
#define EVALUATOR(x, y, z) PASTER(x, y, z)
#define NAME(fun, t1, t2) EVALUATOR(fun, t1, t2)

#define PASTER1(x,y) x ## _ ## y
#define EVALUATOR1(x, y) PASTER1(x, y)
#define NAME1(fun, t1) EVALUATOR1(fun, t1)