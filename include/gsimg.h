#ifndef _GSIMG_H
#define _GSIMG_H

#include <stdio.h>

#include "matrix.h"

typedef struct
{
    int rows, cols, label;
    unsigned char *entries;
} GrayscaleImage;

GrayscaleImage *make_grayscale_image(int rows, int cols);
void destroy_grayscale_image(GrayscaleImage *image);
char get_char_from_grayscale(unsigned char c);
void display_grayscale_image(GrayscaleImage *image);

#endif