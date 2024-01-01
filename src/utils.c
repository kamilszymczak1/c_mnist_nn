#include <math.h>
#include <stdio.h>

#include "utils.h"

void display_progress_bar(double completion, int bar_width, char *description)
{
    printf("\r");
    printf("%s: [", description);

    int full_bars = round(completion * (double)bar_width);

    for (int i = 0; i < full_bars; i++)
        putchar('#');
    for (int i = 0; i < bar_width - full_bars; i++)
        putchar('-');
    printf("] %.2f %%", 100.0f * completion);
    fflush(stdout);
}