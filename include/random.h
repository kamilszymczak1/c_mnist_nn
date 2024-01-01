#ifndef _RANDOM_H
#define _RANDOM_H

void random_init(unsigned long long seed);
unsigned long long random_int64();
unsigned int random_int32();
int random_int(int a, int b);
double random_double();
double random_double_in_range(double a, double b);
double random_normal();
double random_normal_parametrized(double mu, double sigma);

#endif