#include <stdlib.h>

#include "mnistrd.h"
#include "gsimg.h"
#include "matrix.h"

unsigned int read_4_bytes(FILE *fp)
{
    unsigned char buffer[4];
    int bytes_read = fread(buffer, 1, 4, fp);
    if (bytes_read < 4)
    {
        printf("Could not read 4 bytes. Read %d bytes instead\n", bytes_read);
    }
    int result = 0;
    for (int i = 0; i < 4; i++)
        result = (result << 8) | buffer[i];
    return result;
}

unsigned char read_byte(FILE *fp)
{
    unsigned char c;
    if (fread(&c, 1, 1, fp) < 1)
        printf("Could not read a byte\n");
    return c;
}

GrayscaleImage *read_grayscale_image(int rows, int cols, FILE *fp)
{
    GrayscaleImage *img = make_grayscale_image(rows, cols);
    int bytes_read = fread(img->entries, 1, rows * cols, fp);
    if (bytes_read < rows * cols)
    {
        destroy_grayscale_image(img);
        printf("Failed to read the image\n");
        return NULL;
    }
    return img;
}

int read_mnist_images_with_labels(const char *images_path, const char *labels_path, GrayscaleImage ***images)
{
    FILE *images_fp = fopen(images_path, "r");

    if (images_fp == NULL)
    {
        printf("Failed to open %s\n", images_path);
        return 0;
    }

    FILE *labels_fp = fopen(labels_path, "r");

    if (labels_fp == NULL)
    {
        printf("Failed to open %s\n", labels_path);
        fclose(images_fp);
        return 0;
    }

    int images_magic = read_4_bytes(images_fp);
    int images_size = read_4_bytes(images_fp);
    int images_rows = read_4_bytes(images_fp);
    int images_cols = read_4_bytes(images_fp);

    int labels_magic = read_4_bytes(labels_fp);
    int labels_size = read_4_bytes(labels_fp);

    if (images_magic != 2051)
    {
        printf("Magic number mismatch when reading from %s. Got %d, expected 2051\n", images_path, images_magic);
        fclose(labels_fp);
        fclose(images_fp);
        return 0;
    }

    if (labels_magic != 2049)
    {
        printf("Magic number mismatch when reading from %s. Got %d, expected 2049\n", labels_path, labels_magic);
        fclose(labels_fp);
        fclose(images_fp);
        return 0;
    }

    if (labels_size != images_size)
    {
        printf("The number of images does not match the number of labels\n");
        fclose(labels_fp);
        fclose(images_fp);
        return 0;
    }

    *images = (GrayscaleImage **)malloc(sizeof(GrayscaleImage *) * images_size);

    if (images == NULL)
    {
        printf("Failed to allocate memory for images.");
        fclose(labels_fp);
        fclose(images_fp);
        return 0;
    }

    for (int i = 0; i < images_size; i++)
    {
        (*images)[i] = read_grayscale_image(images_rows, images_cols, images_fp);
        (*images)[i]->label = read_byte(labels_fp);
    }

    fclose(labels_fp);
    fclose(images_fp);

    return images_size;
}

void process_gsimg(GrayscaleImage *img, Matrix **in, Matrix **out)
{
    *in = make_matrix(img->rows * img->cols, 1);
    *out = make_matrix(10, 1);

    for (int i = 0; i < img->rows; i++)
        for (int j = 0; j < img->cols; j++)
            (*in)->entries[i * img->cols + j][0] = (double)img->entries[i * img->cols + j] / 255.0f;
    (*out)->entries[img->label][0] = 1.0f;
}