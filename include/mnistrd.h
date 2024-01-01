#ifndef _MNISTRD_H
#define _MNISTRD_H

#include <stdlib.h>

#include "gsimg.h"

unsigned int read_4_bytes(FILE *fp);
unsigned char read_byte(FILE *fp);
int read_mnist_images_with_labels(const char *images_path, const char *labels_path, GrayscaleImage ***images);
GrayscaleImage *read_grayscale_image(int rows, int cols, FILE *fp);
void process_gsimg(GrayscaleImage *img, Matrix **in, Matrix **out);

#endif