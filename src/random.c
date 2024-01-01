#include <math.h>

#include "random.h"

// This implementation comes from the third edition of
// "Numerical Recipes - The Art of Scientific Computing"
// by William H. Press, Saul A. Teukolsky,
// William T. Vetterling and Brian P. Flannery.

const double M_PI = 3.1415926535897932384626433;

unsigned long long __v = 4101842887655102017LL, __w = 1, __u = 0;

double __normal_cached = 0.0f;
int __is_normal_cached = 0;

void random_init(unsigned long long seed)
{
    __u = seed ^ __v;
    random_int64();
    __v = __u;
    random_int64();
    __w = __v;
    random_int64();
}

unsigned long long random_int64()
{
    __u = __u * 2862933555777941757LL + 7046029254386353087LL;
    __v ^= __v >> 17;
    __v ^= __v << 31;
    __v ^= __v >> 8;
    __w = 4294957665U * (__w & 0xffffffff) + (__w >> 32);
    unsigned long long x = __u ^ (__u << 21);
    x ^= x >> 35;
    x ^= x << 4;
    return (x + __v) ^ __w;
}

unsigned int random_int32()
{
    return (unsigned int)random_int64();
}

int random_int(int a, int b)
{
    return a + random_int32() % (b - a + 1);
}

double random_double()
{
    return 5.42101086242752217E-20 * random_int64();
}

double random_double_in_range(double a, double b)
{
    return a + random_double() * (b - a);
}

double random_normal()
{
    if (__is_normal_cached)
    {
        __is_normal_cached = 0;
        return __normal_cached;
    }
    double u1 = random_double();
    double u2 = random_double();
    while (u1 == 0.0f)
        u1 = random_double();
    u1 = sqrt(-2 * log(u1));
    u2 *= 2 * M_PI;

    __normal_cached = u1 * sin(u2);
    __is_normal_cached = 1;
    return u1 * cos(u2);
}

double random_normal_parametrized(double mu, double sigma)
{
    return mu + random_normal() * sigma;
}