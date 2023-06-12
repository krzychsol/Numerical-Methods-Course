#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_monte_plain.h>
#include <gsl/gsl_monte.h>

double fun1(double *k, size_t dim, void *params) {
    double x = k[0];
    return x*x+x+1;
}

double fun2(double *k, size_t dim, void *params) {
    double x = k[0];
    return sqrt(1-x*x);
}

double fun3(double *k, size_t dim, void *params) {
    double x = k[0];
    return 1/sqrt(x);
}

int main(int argc, char* argv[]) {
    int n = atoi(argv[1]);
    size_t dim = 1;
    double res, err;
    double xl[1] = {0};
    double xu[1] = {1};
    gsl_rng *r = gsl_rng_alloc(gsl_rng_taus);
    gsl_monte_plain_state *s = gsl_monte_plain_alloc(dim);

    //f(x) = x^2+x+1
    gsl_monte_function F = {&fun1, dim, NULL};
    gsl_monte_plain_integrate(&F, xl, xu, dim, n, r, s, &res, &err);
    printf("x^2+x+1\n");
    printf("res = %f\n", res);
    printf("err = %f\n", err);

    //f(x) = sqrt(1-x*x)
    gsl_monte_function G = {&fun2, dim, NULL};
    gsl_monte_plain_integrate(&G, xl, xu, dim, n, r, s, &res, &err);
    printf("sqrt(1-x*x)\n");
    printf("res = %f\n", res);
    printf("err = %f\n", err);

    //f(x) = 1/sqrt(x)
    gsl_monte_function H = {&fun3, dim, NULL};
    gsl_monte_plain_integrate(&H, xl, xu, dim, n, r, s, &res, &err);
    printf("1/sqrt(x)\n");
    printf("res = %f\n", res);
    printf("err = %f\n", err);
    
    gsl_monte_plain_free(s);
    gsl_rng_free(r);
    return 0;
}