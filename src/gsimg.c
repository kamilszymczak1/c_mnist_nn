#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "gsimg.h"
#include "matrix.h"

// ------ It's here temporarily -------

void *safe_malloc(size_t size)
{
    void *ptr = malloc(size);
    if (ptr == NULL)
    {
        printf("Failed to allocate memory\n");
        exit(EXIT_FAILURE);
    }
    return ptr;
}

// -------------------------------------

GrayscaleImage *make_grayscale_image(int rows, int cols)
{
    GrayscaleImage *img = malloc(sizeof(GrayscaleImage));
    img->rows = rows;
    img->cols = cols;
    img->label = 0;
    img->entries = (unsigned char *)calloc(rows * cols, sizeof(unsigned char));
    return img;
}

void destroy_grayscale_image(GrayscaleImage *img)
{
    if (img == NULL)
        return;
    free(img->entries);
    free(img);
}

const char *ASCII_GRAYSCALE_CHARS = " .,:;irsxA123hHG#9B8WM";

char get_char_from_grayscale(unsigned char c)
{
    int len = strlen(ASCII_GRAYSCALE_CHARS);
    return ASCII_GRAYSCALE_CHARS[(c * len) / 256];
}

void display_grayscale_image(GrayscaleImage *image)
{
    for (int i = 0; i < image->cols + 2; i++)
        putchar('-');
    putchar('\n');
    for (int i = 0; i < image->rows; i++)
    {
        putchar('|');
        for (int j = 0; j < image->cols; j++)
            putchar(get_char_from_grayscale(image->entries[i * image->rows + j]));
        putchar('|');
        putchar('\n');
    }
    for (int i = 0; i < image->cols + 2; i++)
        putchar('-');
    putchar('\n');
    printf("label: %d\n", image->label);
    for (int i = 0; i < image->cols + 2; i++)
        putchar('-');
    putchar('\n');
}
