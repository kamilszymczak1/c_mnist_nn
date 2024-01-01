#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#include "nnet.h"
#include "matrix.h"
#include "random.h"
#include "utils.h"

double sigmoid(double x)
{
    return 1.0f / (1 + exp(-x));
}

double sigmoid_prime(double x)
{
    x = sigmoid(x);
    return x * (1 - x);
}

NeuralNet *make_neural_net(int layers_count, int *layers_sizes)
{
    assert(layers_count >= 2);

    NeuralNet *net = (NeuralNet *)malloc(sizeof(NeuralNet));

    net->layers_count = layers_count;
    net->layers_sizes = (int *)malloc(sizeof(int) * layers_count);
    memcpy(net->layers_sizes, layers_sizes, sizeof(int) * layers_count);

    net->weights = (Matrix **)malloc(sizeof(Matrix *) * (layers_count - 1));
    net->biases = (Matrix **)malloc(sizeof(Matrix *) * (layers_count - 1));

    for (int i = 0; i < layers_count - 1; i++)
    {
        net->weights[i] = make_random_normal_matrix(layers_sizes[i + 1], layers_sizes[i], 0.0f, 1.0f);
        net->biases[i] = make_random_normal_matrix(layers_sizes[i + 1], 1, 0.0f, 1.0f);
        // net->weights[i] = make_random_uniform_matrix(layers_sizes[i + 1], layers_sizes[i], -1.0f, 1.0f);
        // net->biases[i] = make_random_uniform_matrix(layers_sizes[i + 1], 1, -1.0f, 1.0f);
    }

    return net;
}

void destroy_neural_net(NeuralNet *net)
{
    if (net == NULL)
        return;
    free(net->layers_sizes);
    for (int i = 0; i < net->layers_count - 1; i++)
    {
        destroy_matrix(net->weights[i]);
        destroy_matrix(net->biases[i]);
    }
    free(net);
}

void print_neural_net(NeuralNet *net)
{
    printf("Neural net with %d layers\nLayer sizes: ", net->layers_count);
    for (int i = 0; i < net->layers_count; i++)
        printf("%d ", net->layers_sizes[i]);
    putchar('\n');
    for (int i = 0; i < net->layers_count - 1; i++)
    {
        printf("Weights between layers %d and %d:\n", i, i + 1);
        print_matrix(net->weights[i]);
        printf("Biases for layer %d:\n", i + 1);
        print_matrix(net->biases[i]);
    }
}

Matrix *feedforward(NeuralNet *net, Matrix *in_vec)
{
    assert(in_vec->cols == 1 && in_vec->rows == net->layers_sizes[0]);

    Matrix *cur_vec = copy_matrix(in_vec);

    for (int i = 0; i < net->layers_count - 1; i++)
    {
        Matrix *new_vec = multiply_matrices(net->weights[i], cur_vec);
        add_to_matrix(new_vec, net->biases[i]);
        destroy_matrix(cur_vec);
        cur_vec = new_vec;
        apply_function(cur_vec, sigmoid);
    }

    return cur_vec;
}

void train_with_sgd(NeuralNet *net, Matrix **inputs, Matrix **desired_outputs, int count, int epochs, int mini_batch_size, double learning_rate)
{
    Matrix **mini_batch_in = (Matrix **)malloc(sizeof(Matrix *) * mini_batch_size);
    Matrix **mini_batch_out = (Matrix **)malloc(sizeof(Matrix *) * mini_batch_size);
    for (int epoch = 1; epoch <= epochs; epoch++)
    {
        display_progress_bar((double)epoch / (double)epochs, 50, "Epoch progress");
        for (int i = 0; i < mini_batch_size; i++)
        {
            int j = random_int(0, count - 1);
            mini_batch_in[i] = inputs[j];
            mini_batch_out[i] = desired_outputs[j];
        }

        train_with_mini_batch(net, mini_batch_in, mini_batch_out, mini_batch_size, learning_rate, 1);
    }

    putchar('\n');
}

void train_with_mini_batch(NeuralNet *net, Matrix **inputs, Matrix **desired_outputs, int count, double learning_rate, int epochs)
{
    int n = net->layers_count;
    for (int epoch = 1; epoch <= epochs; epoch++)
    {
        Matrix **dw = (Matrix **)malloc(sizeof(Matrix *) * (n - 1));
        Matrix **db = (Matrix **)malloc(sizeof(Matrix *) * (n - 1));

        for (int i = 0; i < n - 1; i++)
        {
            dw[i] = make_matrix(net->weights[i]->rows, net->weights[i]->cols);
            db[i] = make_matrix(net->biases[i]->rows, net->biases[i]->cols);
        }

        for (int i = 0; i < count; i++)
            backpropagate(net, inputs[i], desired_outputs[i], learning_rate, dw, db);

        for (int i = 0; i < n - 1; i++)
        {
            multiply_by_scalar(dw[i], learning_rate / (1.0f * count));
            multiply_by_scalar(db[i], learning_rate / (1.0f * count));

            subtract_from_matrix(net->weights[i], dw[i]);
            subtract_from_matrix(net->biases[i], db[i]);
        }
    }
}

void backpropagate(NeuralNet *net, Matrix *input, Matrix *desired_output, double learning_rate, Matrix **dw, Matrix **db)
{

    int n = net->layers_count;

    Matrix **is = calloc(n, sizeof(Matrix *)); // Inputs of neurons in each layer
    Matrix **os = calloc(n, sizeof(Matrix *)); // Outputs of neurons in each layer
    os[0] = copy_matrix(input);

    for (int i = 0; i < n - 1; i++)
    {
        is[i + 1] = multiply_matrices(net->weights[i], os[i]);
        add_to_matrix(is[i + 1], net->biases[i]);
        os[i + 1] = copy_matrix(is[i + 1]);
        apply_function(os[i + 1], sigmoid);
    }

    Matrix **ps = calloc(n, sizeof(Matrix *)); // Derivatives of the activation function
                                               // evaluated at inputs of neurons

    for (int i = 1; i < n; i++)
    {
        ps[i] = copy_matrix(is[i]);
        apply_function(ps[i], sigmoid_prime);
    }

    Matrix **ds = calloc(n, sizeof(Matrix *)); // Derivatives of cost function with
                                               // respect to the input of each neuron.

    ds[n - 1] = subtract_matrices(os[n - 1], desired_output);
    hadamard(ds[n - 1], ps[n - 1]);

    for (int i = n - 2; i > 0; i--)
    {
        Matrix *t = transpose(net->weights[i]);
        ds[i] = multiply_matrices(t, ds[i + 1]);
        hadamard(ds[i], ps[i]);
        destroy_matrix(t);
    }

    for (int i = 0; i < n - 1; i++)
    {
        Matrix *my_db = copy_matrix(ds[i + 1]);
        add_to_matrix(db[i], my_db);

        Matrix *os_trans = transpose(os[i]);
        Matrix *my_dw = multiply_matrices(ds[i + 1], os_trans);
        add_to_matrix(dw[i], my_dw);

        destroy_matrix(os_trans);
        destroy_matrix(my_db);
        destroy_matrix(my_dw);
    }

    for (int i = 0; i < n; i++)
    {
        destroy_matrix(is[i]);
        destroy_matrix(os[i]);
        destroy_matrix(ps[i]);
        destroy_matrix(ds[i]);
    }

    free(is);
    free(os);
    free(ps);
    free(ds);
}

void save_neural_net(NeuralNet *net, char *path)
{
    FILE *f = fopen(path, "w");

    fprintf(f, "%d\n", net->layers_count);
    for (int i = 0; i < net->layers_count; i++)
        fprintf(f, "%d ", net->layers_sizes[i]);
    fprintf(f, "\n");

    for (int i = 0; i < net->layers_count - 1; i++)
    {
        for (int j = 0; j < net->layers_sizes[i + 1]; j++)
        {
            for (int k = 0; k < net->layers_sizes[i]; k++)
                fprintf(f, "%.10f ", net->weights[i]->entries[j][k]);
            fprintf(f, "\n");
        }
        for (int j = 0; j < net->layers_sizes[i + 1]; j++)
            fprintf(f, "%.10f ", net->biases[i]->entries[j][0]);
        fprintf(f, "\n");
    }

    fclose(f);
}

NeuralNet *load_neural_net(char *path)
{
    printf("Loading a neural net...\n");
    FILE *f = fopen(path, "r");
    if (f == NULL)
    {
        printf("Failed to open %s\n", path);
        return NULL;
    }
    int layers_count;
    fscanf(f, "%d", &layers_count);
    int *layers_sizes = (int *)malloc(sizeof(int) * layers_count);
    for (int i = 0; i < layers_count; i++)
        fscanf(f, "%d", &layers_sizes[i]);

    printf("Layers count: %d\n", layers_count);
    printf("Layers sizes: ");
    for (int i = 0; i < layers_count; i++)
        printf("%d ", layers_sizes[i]);
    printf("\n");

    NeuralNet *net = make_neural_net(layers_count, layers_sizes);

    for (int i = 0; i < layers_count - 1; i++)
    {
        for (int j = 0; j < layers_sizes[i + 1]; j++)
            for (int k = 0; k < layers_sizes[i]; k++)
                fscanf(f, "%lf", &net->weights[i]->entries[j][k]);

        for (int j = 0; j < layers_sizes[i + 1]; j++)
            fscanf(f, "%lf", &net->biases[i]->entries[j][0]);
    }

    printf("Neural net successfully loaded!\n");

    return net;
}

void validate_net(NeuralNet *net, Matrix **validation_inputs, int *labels, int count)
{
    printf("Validating a neural net...\n");
    int correct = 0;
    for (int i = 0; i < count; i++)
    {
        display_progress_bar((double)i / (double)(count - 1), 50, "Validation progress");
        Matrix *output = feedforward(net, validation_inputs[i]);
        int prediction = 0;
        for (int j = 0; j < 10; j++)
            if (output->entries[j][0] > output->entries[prediction][0])
                prediction = j;

        if (prediction == labels[i])
            correct++;
    }

    printf("\nThe network correctly classfied %d out of %d images (%.2f %%)\n", correct, count, 100.0f * (double)correct / (double)count);
}