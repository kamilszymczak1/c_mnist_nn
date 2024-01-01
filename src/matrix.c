#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <stdio.h>

#include "matrix.h"
#include "random.h"

Matrix *make_matrix(int rows, int cols)
{
    Matrix *m = (Matrix *)malloc(sizeof(Matrix));
    m->rows = rows;
    m->cols = cols;
    m->entries = (double **)malloc(sizeof(double *) * rows);

    for (int i = 0; i < rows; i++)
        m->entries[i] = calloc(cols, sizeof(double));

    return m;
}

Matrix *make_random_uniform_matrix(int rows, int cols, double range_l, double range_r)
{
    Matrix *m = make_matrix(rows, cols);
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            m->entries[i][j] = random_double_in_range(range_l, range_r);
    return m;
}

Matrix *make_random_normal_matrix(int rows, int cols, double mu, double sigma)
{
    Matrix *m = make_matrix(rows, cols);
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            m->entries[i][j] = random_normal_parametrized((double)mu, (double)sigma);
    return m;
}

void destroy_matrix(Matrix *m)
{
    if (m == NULL)
        return;
    for (int i = 0; i < m->rows; i++)
        free(m->entries[i]);
    free(m->entries);
    free(m);
}

Matrix *copy_matrix(Matrix *m)
{
    Matrix *m_cpy = make_matrix(m->rows, m->cols);
    for (int i = 0; i < m->rows; i++)
        memcpy(m_cpy->entries[i], m->entries[i], sizeof(double) * m->cols);
    return m_cpy;
}

void print_matrix(Matrix *m)
{
    printf("Matrix of size %dx%d:\n", m->rows, m->cols);
    for (int i = 0; i < m->rows; i++)
    {
        for (int j = 0; j < m->cols; j++)
            printf("%.5f ", m->entries[i][j]);
        putchar('\n');
    }
}

void apply_function(Matrix *m, double (*f)(double))
{
    for (int i = 0; i < m->rows; i++)
        for (int j = 0; j < m->cols; j++)
            m->entries[i][j] = f(m->entries[i][j]);
}

Matrix *add_matrices(Matrix *a, Matrix *b)
{
    assert(a->cols == b->cols);
    assert(a->rows == b->rows);

    Matrix *c = make_matrix(a->rows, a->cols);

    for (int i = 0; i < a->rows; i++)
        for (int j = 0; j < a->cols; j++)
            c->entries[i][j] = a->entries[i][j] + b->entries[i][j];
    return c;
}

void *add_to_matrix(Matrix *a, Matrix *b)
{
    assert(a->cols == b->cols);
    assert(a->rows == b->rows);

    for (int i = 0; i < a->rows; i++)
        for (int j = 0; j < a->cols; j++)
            a->entries[i][j] += b->entries[i][j];
}

Matrix *subtract_matrices(Matrix *a, Matrix *b)
{
    assert(a->cols == b->cols);
    assert(a->rows == b->rows);

    Matrix *c = make_matrix(a->rows, a->cols);

    for (int i = 0; i < a->rows; i++)
        for (int j = 0; j < a->cols; j++)
            c->entries[i][j] = a->entries[i][j] - b->entries[i][j];
    return c;
}

void *subtract_from_matrix(Matrix *a, Matrix *b)
{
    assert(a->cols == b->cols);
    assert(a->rows == b->rows);

    for (int i = 0; i < a->rows; i++)
        for (int j = 0; j < a->cols; j++)
            a->entries[i][j] -= b->entries[i][j];
}

Matrix *multiply_matrices(Matrix *a, Matrix *b)
{
    assert(a->cols == b->rows);

    Matrix *c = make_matrix(a->rows, b->cols);
    for (int i = 0; i < a->rows; i++)
        for (int j = 0; j < b->cols; j++)
            for (int k = 0; k < a->cols; k++)
                c->entries[i][j] += a->entries[i][k] * b->entries[k][j];
    return c;
}

void hadamard(Matrix *a, Matrix *b)
{
    assert(a->cols == b->cols);
    assert(a->rows == b->rows);

    for (int i = 0; i < a->rows; i++)
        for (int j = 0; j < a->cols; j++)
            a->entries[i][j] *= b->entries[i][j];
}

Matrix *transpose(Matrix *a)
{
    Matrix *m = make_matrix(a->cols, a->rows);
    for (int i = 0; i < a->rows; i++)
        for (int j = 0; j < a->cols; j++)
            m->entries[j][i] = a->entries[i][j];
    return m;
}

void multiply_by_scalar(Matrix *a, double x)
{
    for (int i = 0; i < a->rows; i++)
        for (int j = 0; j < a->cols; j++)
            a->entries[i][j] *= x;
}

double squared_frobenius_norm(Matrix *a)
{
    double x = 0.0f;
    for (int i = 0; i < a->rows; i++)
        for (int j = 0; j < a->cols; j++)
            x += a->entries[i][j] * a->entries[i][j];
    return x;
}
