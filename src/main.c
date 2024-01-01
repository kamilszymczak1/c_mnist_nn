#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>

#include "matrix.h"
#include "gsimg.h"
#include "random.h"
#include "nnet.h"
#include "mnistrd.h"

const char *TRAINING_IMAGES_PATH = "./data/train-images-idx3-ubyte";
const char *TRAINING_LABELS_PATH = "./data/train-labels-idx1-ubyte";
const char *VALIDATION_IMAGES_PATH = "./data/t10k-images-idx3-ubyte";
const char *VALIDATION_LABELS_PATH = "./data/t10k-labels-idx1-ubyte";

void create_nn(NeuralNet **net, char current_path[100])
{
    printf("Input the number of layers: ");
    int layers_count;
    scanf("%d", &layers_count);
    printf("Input the layer's sizes (including the input and output layers): ");
    int *layers_sizes = (int *)malloc(sizeof(int) * layers_count);

    for (int i = 0; i < layers_count; i++)
        scanf("%d", &layers_sizes[i]);

    printf("Initializing the net...\n");
    *net = make_neural_net(layers_count, layers_sizes);
    strcpy(current_path, "untitled");
    printf("The neural net has been successfully created\n");
}

void load_nn(NeuralNet **net, char current_path[64])
{
    destroy_neural_net(*net);
    printf("Loading a neural network. Enter the path: ");
    scanf("%63s", current_path);
    *net = load_neural_net(current_path);
}

void save_nn(NeuralNet **net, char current_path[64])
{
    printf("Saving the current net. Enter the path: ");
    scanf("%63s", current_path);
    save_neural_net(*net, current_path);
}

void train_nn(NeuralNet **net, char current_path[64], Matrix **inputs, Matrix **outputs, int count)
{
    double learning_rate;
    int batch_size, epochs;
    printf("Enter the learning rate: ");
    scanf("%lf", &learning_rate);
    printf("Enter the batch size: ");
    scanf("%d", &batch_size);
    printf("Enter the number of epochs: ");
    scanf("%d", &epochs);
    printf("Training the neural net...\n");
    train_with_sgd(*net, inputs, outputs, count, epochs, batch_size, learning_rate);
}

void validate_nn(NeuralNet **net, char current_path[64], Matrix **inputs, int *labels, int count)
{
    validate_net(*net, inputs, labels, count);
}

int main()
{
    random_init(1);

    printf("Starting MNIST Neural Network\n");

    GrayscaleImage **training_images = NULL, **validation_images = NULL;
    int training_images_size = read_mnist_images_with_labels(TRAINING_IMAGES_PATH, TRAINING_LABELS_PATH, &training_images);
    int validation_images_size = read_mnist_images_with_labels(VALIDATION_IMAGES_PATH, VALIDATION_LABELS_PATH, &validation_images);

    printf("Number of training images: %d\n", training_images_size);
    printf("Number of validation images: %d\n", validation_images_size);

    Matrix **training_inputs = (Matrix **)malloc(sizeof(Matrix *) * training_images_size);
    Matrix **training_outputs = (Matrix **)malloc(sizeof(Matrix *) * training_images_size);

    for (int i = 0; i < training_images_size; i++)
        process_gsimg(training_images[i], &training_inputs[i], &training_outputs[i]);

    Matrix **validation_inputs = (Matrix **)malloc(sizeof(Matrix *) * validation_images_size);
    Matrix **validation_outputs = (Matrix **)malloc(sizeof(Matrix *) * validation_images_size);

    for (int i = 0; i < validation_images_size; i++)
        process_gsimg(validation_images[i], &validation_inputs[i], &validation_outputs[i]);

    int *labels = (int *)malloc(sizeof(int) * validation_images_size);
    for (int i = 0; i < validation_images_size; i++)
        labels[i] = validation_images[i]->label;

    bool running = true;

    NeuralNet *net = NULL;
    char current_path[64];
    for (int i = 0; i < 64; i++)
        current_path[i] = 0;

    while (running)
    {
        printf("\nCurrent neural net: %s\n", strlen(current_path) > 0 ? current_path : "(None)");

        printf("What would you like to do?\n1: Create a new neural net\n2: Load a neural net\n3: Save the current net\n4: Train\n5: Validate\n6: Exit\nEnter you choice: ");

        int choice;
        scanf("%d", &choice);

        switch (choice)
        {
        case 1:
            create_nn(&net, current_path);
            break;
        case 2:
            load_nn(&net, current_path);
            break;
        case 3:
            save_nn(&net, current_path);
            break;
        case 4:
            train_nn(&net, current_path, training_inputs, training_outputs, training_images_size);
            break;
        case 5:
            validate_nn(&net, current_path, validation_inputs, labels, validation_images_size);
            break;
        case 6:
            printf("Exiting...\n");
            running = false;
            break;
        default:
            printf("Incorrect choice!\n");
            break;
        }
    }

    destroy_neural_net(net);

    for (int i = 0; i < training_images_size; i++)
    {
        destroy_matrix(training_inputs[i]);
        destroy_matrix(training_outputs[i]);
    }

    free(training_inputs);
    free(training_outputs);

    for (int i = 0; i < training_images_size; i++)
        destroy_grayscale_image(training_images[i]);
    for (int i = 0; i < validation_images_size; i++)
        destroy_grayscale_image(validation_images[i]);
    free(validation_images);
    free(training_images);

    return 0;
}