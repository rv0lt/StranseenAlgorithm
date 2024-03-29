#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <omp.h>

void matrix_sum(double **a, double **b, double **result, int tam, int nthreads);
void matrix_subtract(double **a, double **b, double **result, int tam, int nthreads);
double **allocate_real_matrix(int tam);
double **free_real_matrix(double **v, int tam);
void strassen(double **A, double **B, double **C, int n, int base, int nthreads);
void naive_multiplication(double **A, double **B, double **C, int n, int nthreads);
int compare_double_matrixes(double **M1, double **M2, int n, double precission, int nthreads);

/*
Classic matrix multiplication
C = A*B
*/
void naive_multiplication(double **A, double **B, double **C, int n, int nthreads)
{
    int i,j;
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            C[i][j] = 0;
            int k;
            for (k = 0; k < n; k++)
            {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}
/*------------------------------------------------------------------------------*/
/*
Algorithm:
   Matrices A and B are split into four smaller
   (N/2)x(N/2) matrices as follows:
          _    _          _   _
     A = | A11  A12 |    B = | B11 B12 |
         | A21  A22 |        | B21 B22 |
          -    -          -   -
   Then we build the following 7 matrices (requiring
   seven (N/2)x(N/2) matrix multiplications:

     M1 = (A11+A22)*(B11 + B22)
     M2 = (A21 + A22)*B11
     M3 = A11 * (B12 - B22)
     M4 = A22*(B21 - B11);
     M5 = (A11 + A12)*B22
     M6 = (A21 - A11)*(B11 + B12)
     M7 = (A12 - A22)*(B21 + B22)

   The final result is
        _                                            _
   C = | M1 + M4 - M5 + M7)   M3 + M5              |
       | P2 + M4                 M1 - M2 + M3 + M6 |
        -                                            -

*/
void strassen(double **A, double **B, double **C, int n, int base, int nthreads)
{

    /*
    Recursive base case.
    If matrices are smaller than base we just use
    the naive algorithm.

    The best choice for base will depend on the hardware used

    Based on:
      https://stackoverflow.com/questions/11495723/why-is-strassen-matrix-multiplication-so-much-slower-than-standard-matrix-multip
    */

    if (n <= base)
    {
        naive_multiplication(A, B, C, n, nthreads);
        return;
    }
    else
    {
        int k = n / 2;   // new matrix size
        double **x, **y; // auxiliary matrixes to store temporary results

        /*new matrixes according to the algorithm*/
        double **m1, **m2, **m3, **m4, **m5, **m6, **m7;
        double **a11, **a12, **a21, **a22;
        double **b11, **b12, **b21, **b22;
        double **c11, **c12, **c21, **c22;

        /* memory allocation: */
        x = allocate_real_matrix(k);
        y = allocate_real_matrix(k);

        m1 = allocate_real_matrix(k);
        m2 = allocate_real_matrix(k);
        m3 = allocate_real_matrix(k);
        m4 = allocate_real_matrix(k);
        m5 = allocate_real_matrix(k);
        m6 = allocate_real_matrix(k);
        m7 = allocate_real_matrix(k);

        a11 = allocate_real_matrix(k);
        a12 = allocate_real_matrix(k);
        a21 = allocate_real_matrix(k);
        a22 = allocate_real_matrix(k);

        b11 = allocate_real_matrix(k);
        b12 = allocate_real_matrix(k);
        b21 = allocate_real_matrix(k);
        b22 = allocate_real_matrix(k);

        c11 = allocate_real_matrix(k);
        c12 = allocate_real_matrix(k);
        c21 = allocate_real_matrix(k);
        c22 = allocate_real_matrix(k);

        int i, j;
        // dividing the matrices in 4 sub-matrices:
        for (i = 0; i < k; i++)
        {
            for (j = 0; j < k; j++)
            {
                a11[i][j] = A[i][j];
                a12[i][j] = A[i][j + k];
                a21[i][j] = A[i + k][j];
                a22[i][j] = A[i + k][j + k];

                b11[i][j] = B[i][j];
                b12[i][j] = B[i][j + k];
                b21[i][j] = B[i + k][j];
                b22[i][j] = B[i + k][j + k];
            }
        }

        /*M1=(A[0][0]+A[1][1])*(B[0][0]+B[1][1])*/
        matrix_sum(a11, a22, x, k, nthreads);  // a11 + a22
        matrix_sum(b11, b22, y, k, nthreads);  // b11 + b22
        strassen(x, y, m1, k, base, nthreads); // (a11+a22) * (b11+b22)

        /*M2=(A[1][0]+A[1][1])*B[0][0]*/
        matrix_sum(a21, a22, x, k, nthreads);    // a21 + a22
        strassen(x, b11, m2, k, base, nthreads); // (a21+a22) * (b11)

        /*M3=A[0][0]*(B[0][1]-B[1][1])*/
        matrix_subtract(b12, b22, x, k, nthreads); // b12 - b22
        strassen(a11, x, m3, k, base, nthreads);   // (a11) * (b12 - b22)

        /*M4=A[1][1]*(B[1][0]-B[0][0])*/
        matrix_subtract(b21, b11, x, k, nthreads); // b21 - b11
        strassen(a22, x, m4, k, base, nthreads);   // (a22) * (b21 - b11)

        /*M5=(A[0][0]+A[0][1])*B[1][1]*/
        matrix_sum(a11, a12, x, k, nthreads);    // a11 + a12
        strassen(x, b22, m5, k, base, nthreads); // (a11+a12) * (b22)

        /*M6=(A[1][0]-A[0][0])*(B[0][0]+B[0][1])*/
        matrix_subtract(a21, a11, x, k, nthreads); // a21 - a11
        matrix_sum(b11, b12, y, k, nthreads);      // b11 + b12
        strassen(x, y, m6, k, base, nthreads);     // (a21-a11) * (b11+b12)

        /*M7=(A[0][1]-A[1][1])*(B[1][0]+B[1][1])*/
        matrix_subtract(a12, a22, x, k, nthreads); // a12 - a22
        matrix_sum(b21, b22, y, k, nthreads);      // b21 + b22
        strassen(x, y, m7, k, base, nthreads);     //  (a12-a22) * (b21+b22)

        /*Calculating the 4 parts for the result matrix*/
        matrix_sum(m3, m5, c12, k, nthreads); // c12 = m3 + m5
        matrix_sum(m2, m4, c21, k, nthreads); // c21 = m2 + m4

        matrix_sum(m1, m4, x, k, nthreads);       // m1 + m4
        matrix_sum(x, m7, y, k, nthreads);        // m1 + m4 + m7
        matrix_subtract(y, m5, c11, k, nthreads); // c11 = m1 + m4 - m5 + m7

        matrix_sum(m1, m3, x, k, nthreads);       // m1 + m3
        matrix_sum(x, m6, y, k, nthreads);        // m1 + m3 + m6
        matrix_subtract(y, m2, c22, k, nthreads); // c22 = m1 + m3 - m2 + m6

        /* Grouping the parts obtained in the result matrix:*/
        for (i = 0; i < k; i++)
        {
            for (j = 0; j < k; j++)
            {
                C[i][j] = c11[i][j];
                C[i][j + k] = c12[i][j];
                C[i + k][j] = c21[i][j];
                C[i + k][j + k] = c22[i][j];
            }
        }

        // deallocating memory (free):
        a11 = free_real_matrix(a11, k);
        a12 = free_real_matrix(a12, k);
        a21 = free_real_matrix(a21, k);
        a22 = free_real_matrix(a22, k);

        b11 = free_real_matrix(b11, k);
        b12 = free_real_matrix(b12, k);
        b21 = free_real_matrix(b21, k);
        b22 = free_real_matrix(b22, k);

        c11 = free_real_matrix(c11, k);
        c12 = free_real_matrix(c12, k);
        c21 = free_real_matrix(c21, k);
        c22 = free_real_matrix(c22, k);

        m1 = free_real_matrix(m1, k);
        m2 = free_real_matrix(m2, k);
        m3 = free_real_matrix(m3, k);
        m4 = free_real_matrix(m4, k);
        m5 = free_real_matrix(m5, k);
        m6 = free_real_matrix(m6, k);
        m7 = free_real_matrix(m7, k);
        x = free_real_matrix(x, k);
        y = free_real_matrix(y, k);
    } // end of else

} // end of Strassen function
/*------------------------------------------------------------------------------*/
/*

Function to compare two double matrixes
precission defines the number of precission that is acceptable

return 0 if not equal
*/
int compare_double_matrixes(double **M1, double **M2, int n, double precission, int nthreads)
{
    int i,j;
    for (i = 0; i < n; i++)
        for (j = 0; j < n; j++)
            if (fabs(M1[i][j] - M2[i][j]) >= precission)
                return 0;
    return 1;
}
/*------------------------------------------------------------------------------*/
/*
C = A+B
*/
void matrix_sum(double **A, double **B, double **C, int n, int nthreads)
{
    int i,j;
    for (i = 0; i < n; i++)
        for (j = 0; j < n; j++)
            C[i][j] = A[i][j] + B[i][j];
}
/*------------------------------------------------------------------------------*/
/*
C = A-B
*/
void matrix_subtract(double **A, double **B, double **C, int n, int nthreads)
{
    int i,j;
    for (i = 0; i < n; i++)
        for (j = 0; j < n; j++)
            C[i][j] = A[i][j] - B[i][j];
}
/*------------------------------------------------------------------------------*/
/*
returns a pointer to a double matrix of size tam
*/
double **allocate_real_matrix(int tam)
{

    int i;
    double **v; // pointer to the vector

    // allocates one vector of vectors (matrix)
    v = (double **)malloc(tam * sizeof(double *));

    if (v == NULL)
    {
        printf("** Error in matrix allocation: insufficient memory **");
        return (NULL);
    }

    // allocates each row of the matrix
    for (i = 0; i < tam; i++)
    {
        v[i] = (double *)malloc(tam * sizeof(double));

        if (v[i] == NULL)
        {
            printf("** Error: Insufficient memory **");
            free_real_matrix(v, tam);
            return (NULL);
        }
    }

    return (v); // returns the pointer to the vector.
}
/*------------------------------------------------------------------------------*/
/*
free the matrix allocated
*/
double **free_real_matrix(double **v, int tam)
{

    int i;

    if (v == NULL)
    {
        return (NULL);
    }

    for (i = 0; i < tam; i++)
    {
        if (v[i])
        {
            free(v[i]); // frees a row of the matrix
            v[i] = NULL;
        }
    }

    free(v); // frees the pointer /
    v = NULL;

    return (NULL); // returns a null pointer /
}
/*------------------------------------------------------------------------------*/
/*
==================================================================================
MAIN
==================================================================================
*/
int main(int argc, char **argv)
{

    if (argc != 5)
    {
        printf("This program calculates a matrix-matrix multiplication using the Stranssen algorithm and compares the result against a Traditional(naive) implementation\n"
               "\nUSE: ./stranssen INTERACTIVE ORDER VERBOSE BASE_CASE\n\n"
               "If INTERACTIVE is set to 0, then random matrixes will be generated, in other case, it will ask the user for input\n"
               "ORDER specifies the order of the matrixes to multiply, only powers of 2 are accepted\n"
               "VERBOSE, Set between 0 and 3 for different levels of verbose output\n"
               "BASE_CASE for the recursion in Strassen\n");
        return 0;
    }

    int interactive = atoi(argv[1]);
    int n = atoi(argv[2]);
    int verbose = atoi(argv[3]);
    int base = atoi(argv[4]);
    int nthreads = 0; // NOT implemented

    int i, j, aux;
    double **A, **B, **C1, **C2;

    /*
    bitwise operation to check if it is a power of 2

    If the number is a power of two, then only 1 bit will be set in its binary representation.
    If we subtract 1 from a number which is power of 2, then all the bits after the set-bit will become set and the set bit will be unset

    N    = 4,  000100
    N-1  = 3,  000011
    N    = 16, 010000
    N-1  = 15, 001111
    */
    if (((n & (n - 1)) != 0) || (n == 0))
    {
        printf("Please use an order that is a power of 2\n");
        return -1;
    }
    if (base <= 0 || base > n)
    {
        printf("Please specify a valid Base case\n");
        return -1;
    }
    if (verbose < 0 || verbose > 3)
    {
        printf("Set a valid verbose level\n");
        return -1;
    }
    /*
    MEMORY ALLOCATION
    */
    A = allocate_real_matrix(n);
    B = allocate_real_matrix(n);
    C1 = allocate_real_matrix(n);
    C2 = allocate_real_matrix(n);

    /*
    ASKING FOR THE INPUT MATRIXES
    */
    if (interactive)
    {
        printf("\nNow enter the first matrix:\n\n");
        for (i = 0; i < n; i++)
        {
            printf("Enter the elements of the %d-th row:", i + 1);
            for (j = 0; j < n; j++)
                scanf(" %lf", &A[i][j]);
        }
        printf("\nNow enter the second matrix:\n\n");
        for (i = 0; i < n; i++)
        {
            printf("Enter the elements of the %d-th row:", i + 1);
            for (j = 0; j < n; j++)
                scanf(" %lf", &B[i][j]);
        }
    } // interactive
    /*
    GENERATE RANDOM INPUT
    */
    else // random input
    {
        /* Intializes random number generator */
        srand(time(0));
        double div = RAND_MAX / 100;
        for (i = 0; i < n; i++)
        {
            for (j = 0; j < n; j++)
            {
                A[i][j] = rand() / div;
                B[i][j] = rand() / div;
            }
        }
    }

    /*
    PRINT INPUT MATRIXES
    */
    if (verbose == 3)
    {
        printf(
            "\n===============================\n"
            "This is the first matrix:\n"
            "===============================\n");
        for (i = 0; i < n; i++)
        {
            printf("\n\n\n");
            for (j = 0; j < n; j++)
                printf("\t%lf", A[i][j]);
        }

        printf(
            "\n\n===============================\n\n\n"
            "This is the second matrix:\n"
            "===============================\n");

        for (i = 0; i < n; i++)
        {
            printf("\n\n\n");
            for (j = 0; j < n; j++)
                printf("\t%lf", B[i][j]);
        }
        printf(
            "\n\n===============================\n\n\n");

    } // Verbose

    /*
    ================================================================
    CALL TO THE ALGORITHMS
    AND CHECK THEIR TIMMINGS
    =================================================================
    */
    double start, end;
    start = omp_get_wtime();
    strassen(A, B, C1, n, base, nthreads);
    end = omp_get_wtime();
    if (verbose >= 2)
        printf("Strassen(s): %f\n", end - start);
    else if (verbose == 1)
        printf("%f\n",end-start);

    start = omp_get_wtime();
    naive_multiplication(A, B, C2, n, nthreads);
    end = omp_get_wtime();
    if (verbose >= 2)
        printf("Naive(s): %f\n", end - start);
    else if (verbose == 1)
        printf("%f\n",end-start);
    /*
    CHECK RESULTS
    */
    aux = compare_double_matrixes(C1, C2, n, 0.00001, nthreads);
    if (!aux)
    {
        printf("\nNO OK\n");
        if (verbose >= 2)
            printf("Check the strassen implementation!!!\n");
        return -1;
    }
    else
    {
        printf("\nOK\n");
        if (verbose >= 2)
            printf("Strassen and naive algorithm yield same results! Yay!\n");
    }
    /*
    ========================================================================
    */

    /*
    PRINT THE RESULT MATRIX
    */
    if (verbose == 3)
    {
        printf(
            "\n\n===============================\n\n\n"
            "This is the final matrix:\n"
            "===============================\n");
        for (i = 0; i < n; i++)
        {
            printf("\n\n\n");
            for (j = 0; j < n; j++)
                printf("\t%lf", C1[i][j]);
        }
        printf("\n");
    } // print result

    /*
    TODO:
        study and parallelize (openmp, valgrind)
        measure results to choice best BASE_CASE and N_THREADS
        optimize code
    */

    /*
    CLEANUP
    */
    free_real_matrix(A, n);
    free_real_matrix(B, n);
    free_real_matrix(C1, n);
    free_real_matrix(C2, n);

    return 0;
}
