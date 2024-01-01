#ifndef _MATRIX_H
#define _MATRIX_H

typedef struct
{
    int rows, cols;
    double **entries;
} Matrix;

Matrix *make_matrix(int rows, int cols);
Matrix *make_random_uniform_matrix(int rows, int cols, double range_l, double range_r);
Matrix *make_random_normal_matrix(int rows, int cols, double mu, double sigma);
void destroy_matrix(Matrix *m);
void print_matrix(Matrix *m);
void apply_function(Matrix *m, double (*f)(double));
Matrix *copy_matrix(Matrix *m);
Matrix *add_matrices(Matrix *a, Matrix *b);
void *add_to_matrix(Matrix *a, Matrix *b);
Matrix *subtract_matrices(Matrix *a, Matrix *b);
void *subtract_from_matrix(Matrix *a, Matrix *b);
Matrix *multiply_matrices(Matrix *a, Matrix *b);
void hadamard(Matrix *a, Matrix *b);
Matrix *transpose(Matrix *a);
void multiply_by_scalar(Matrix *a, double r);
double squared_frobenius_norm(Matrix *a);

#endif