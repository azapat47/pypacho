inline void __attribute__((overloadable)) Atomic_Add(volatile __global float *addr, float val)
{
  union {
    unsigned int u32;
    float        f32;
  } next, expected, current;
  current.f32    = *addr;
  do {
    expected.f32 = current.f32;
    next.f32     = expected.f32 + val;
    current.u32  = atomic_cmpxchg( (volatile __global unsigned int *)addr, 
				   expected.u32, next.u32);
  } while( current.u32 != expected.u32 );
}

inline void __attribute__((overloadable)) Atomic_Add (__global double *val, double delta) {
  union {
    double f;
    ulong  i;
  } old;
  union {
    double f;
    ulong  i;
  } new;
  do {
    old.f = *val;
    new.f = old.f + delta;
  } while (atom_cmpxchg ( (volatile __global ulong *)val, old.i, new.i) != old.i);
}

inline void __attribute__((overloadable)) Atomic_Add(__global int *addr, int val) {
  atomic_add(addr, val);
}