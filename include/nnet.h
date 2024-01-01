#ifndef _NNET_H
#define _NNET_H

#include "matrix.h"

typedef struct
{
    int layers_count, *layers_sizes;
    Matrix **weights, **biases;
} NeuralNet;

NeuralNet *
make_neural_net(int layers_count, int *layers_sizes);
void destroy_neural_net(NeuralNet *net);
void print_neural_net(NeuralNet *net);
Matrix *feedforward(NeuralNet *net, Matrix *in_vec);
void train_with_sgd(NeuralNet *net, Matrix **inputs, Matrix **desired_outputs, int count, int epochs, int mini_batch_size, double learning_rate);
void train_with_mini_batch(NeuralNet *net, Matrix **inputs, Matrix **desired_outputs, int count, double learning_rate, int epochs);
void backpropagate(NeuralNet *net, Matrix *input, Matrix *desired_output, double learning_rate, Matrix **dw, Matrix **db);
void save_neural_net(NeuralNet *net, char *path);
NeuralNet *load_neural_net(char *path);
void validate_net(NeuralNet *net, Matrix **validation_inputs, int *labels, int count);

#endif