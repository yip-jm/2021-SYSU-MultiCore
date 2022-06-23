#ifndef _INCL_CONFIG
#define _INCL_CONFIG
#include "utils.h"

/*
 * Don't change the settings below if you are going to use the rexec utility
 */

// Load a matrix from [stream] into [sample].
bool loadMatrix(FILE* stream, int* width, int* height, int** sample) {
    if (fread(width, sizeof(int), 1, stream) == 0) {
        // eof
        return false;
    }
    fread(height, sizeof(int), 1, stream);
    int size = (*width) * (*height);

    *sample = (int*)malloc(sizeof(int) * size);
    if (*sample == NULL) {
        printf("failed to allocate\n");
        return false;
    }

    fread(*sample, sizeof(int), size, stream);

    return true;
}

// Save matrix [sample] into [stream] .
void saveMatrix(FILE* stream, int* width, int* height, float** sample) {
    int size = (*width) * (*height);
    W_CHK(fwrite(width, sizeof(int), 1, stream));
    W_CHK(fwrite(height, sizeof(int), 1, stream));
    W_CHK(fwrite(*sample, sizeof(float), size, stream));
}

// The relative path to the data file
char *inputPath = "./input1.bin";
// The relative path to the output results
char *outputPath = "./output1.bin";

#endif