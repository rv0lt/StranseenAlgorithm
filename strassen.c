#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// https://en.wikipedia.org/wiki/Strassen_algorithm
void Msum(double **a, double **b, double **result, int tam);
void Msubtract(double **a, double **b, double **result, int tam);
double **allocate_real_matrix(int tam);
double **free_real_matrix(double **v, int tam);
void strassen(double **A, double **B, double **C, int n);

void strassen(double **A, double **B, double **C, int n)
{

    // trivial cases ( matrix is 1 X 1)
    if (n == 1)
    {
        C[0][0] = A[0][0] * B[0][0];
        return;
    }
    else if (n == 2) //( matrix is 2 X 2)
    {
        double m1 = (A[0][0] + A[1][1]) * (B[0][0] + B[1][1]); // M1=(A[0][0]+A[1][1])*(B[0][0]+B[1][1])
        double m2 = (A[1][0] + A[1][1]) * B[0][0];             // M2=(A[1][0]+A[1][1])*B[0][0]
        double m3 = A[0][0] * (B[0][1] - B[1][1]);             // M3=A[0][0]*(B[0][1]-B[1][1])
        double m4 = A[1][1] * (B[1][0] - B[0][0]);             // M4=A[1][1]*(B[1][0]-B[0][0])
        double m5 = (A[0][0] + A[0][1]) * B[1][1];             // M5=(A[0][0]+A[0][1])*B[1][1]
        double m6 = (A[1][0] - A[0][0]) * (B[0][0] + B[0][1]); // M6=(A[1][0]-A[0][0])*(B[0][0]+B[0][1])
        double m7 = (A[0][1] - A[1][1]) * (B[1][0] + B[1][1]); // M7=(A[0][1]-A[1][1])*(B[1][0]+B[1][1])

        C[0][0] = m1 + m4 - m5 + m7;
        C[0][1] = m3 + m5;
        C[1][0] = m2 + m4;
        C[1][1] = m1 + m3 - m2 + m6;
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
        Msum(a11, a22, x, k);  // a11 + a22
        Msum(b11, b22, y, k);  // b11 + b22
        strassen(x, y, m1, k); // (a11+a22) * (b11+b22)

        /*M2=(A[1][0]+A[1][1])*B[0][0]*/
        Msum(a21, a22, x, k);    // a21 + a22
        strassen(x, b11, m2, k); // (a21+a22) * (b11)

        /*M3=A[0][0]*(B[0][1]-B[1][1])*/
        Msubtract(b12, b22, x, k); // b12 - b22
        strassen(a11, x, m3, k);   // (a11) * (b12 - b22)

        /*M4=A[1][1]*(B[1][0]-B[0][0])*/
        Msubtract(b21, b11, x, k); // b21 - b11
        strassen(a22, x, m4, k);   // (a22) * (b21 - b11)

        /*M5=(A[0][0]+A[0][1])*B[1][1]*/
        Msum(a11, a12, x, k);    // a11 + a12
        strassen(x, b22, m5, k); // (a11+a12) * (b22)

        /*M6=(A[1][0]-A[0][0])*(B[0][0]+B[0][1])*/
        Msubtract(a21, a11, x, k); // a21 - a11
        Msum(b11, b12, y, k);      // b11 + b12
        strassen(x, y, m6, k);     // (a21-a11) * (b11+b12)

        /*M7=(A[0][1]-A[1][1])*(B[1][0]+B[1][1])*/
        Msubtract(a12, a22, x, k); // a12 - a22
        Msum(b21, b22, y, k);      // b21 + b22
        strassen(x, y, m7, k);     //  (a12-a22) * (b21+b22)

        /*Calculating the 4 parts for the result matrix*/
        Msum(m3, m5, c12, k); // c12 = m3 + m5
        Msum(m2, m4, c21, k); // c21 = m2 + m4

        Msum(m1, m4, x, k);       // m1 + m4
        Msum(x, m7, y, k);        // m1 + m4 + m7
        Msubtract(y, m5, c11, k); // c11 = m1 + m4 - m5 + m7

        Msum(m1, m3, x, k);       // m1 + m3
        Msum(x, m6, y, k);        // m1 + m3 + m6
        Msubtract(y, m2, c22, k); // c22 = m1 + m3 - m2 + m6

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

void Msum(double **A, double **B, double **C, int n)
{
    int i, j;
    for (i = 0; i < n; i++)
        for (j = 0; j < n; j++)
            C[i][j] = A[i][j] + B[i][j];
}
/*------------------------------------------------------------------------------*/

void Msubtract(double **A, double **B, double **C, int n)
{
    int i, j;
    for (i = 0; i < n; i++)
        for (j = 0; j < n; j++)
            C[i][j] = A[i][j] - B[i][j];
}
/*------------------------------------------------------------------------------*/

double **allocate_real_matrix(int tam)
{

    int i, j;
    double **v, a; // pointer to the vector

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

int main(int argc, char **argv)
{

    if (argc != 3)
    {
        printf("\nUSE: ./stranssen INTERACTIVE ORDER\n If interactive is set to 0, then random matrixes will be generated, in other case, it will ask the user for input\n ORDER specifies the order of the matrixes to multiply\n");
        return 0;
    }
    int i, j;
    double **A, **B, **C;
    int it = atoi(argv[1]);
    int n = atoi(argv[2]);

    A = allocate_real_matrix(n);
    B = allocate_real_matrix(n);
    C = allocate_real_matrix(n);

    if (it)
    {
        printf("\nNow enter the first matrix:\n\n");
        for (i = 0; i < n; i++)
        {
            printf("Enter the elements of the %d-th row:", i + 1);
            for (j = 0; j < n; j++)
                scanf(" %lf", &A[i][j]);
        }

        /*Input second matrix*/
        printf("\nNow enter the second matrix:\n\n");
        for (i = 0; i < n; i++)
        {
            printf("Enter the elements of the %d-th row:", i + 1);
            for (j = 0; j < n; j++)
                scanf(" %lf", &B[i][j]);
        }
    }
    else
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

    /*Printing first matrix*/
    printf("\nThis is the first matrix:");
    for (i = 0; i < n; i++)
    {
        printf("\n\n\n");
        for (j = 0; j < n; j++)
            printf("\t%lf", A[i][j]);
    }
    /*Printing second matrix*/
    printf("\n\n\nThis is the second matrix:");
    for (i = 0; i < n; i++)
    {
        printf("\n\n\n");
        for (j = 0; j < n; j++)
            printf("\t%lf", B[i][j]);
    }

    strassen(A, B, C, n); // Calling the function.

    /*Printing the final matrix*/
    printf("\n\n\nThis is the final matrix:");
    for (i = 0; i < n; i++)
    {
        printf("\n\n\n");
        for (j = 0; j < n; j++)
            printf("\t%lf", C[i][j]);
    }
    printf("\n");

    /*
    TODO:
        check results with a naive implementation
        study and parallelize (openmp valgrind)
        measure results
        make code faster
    */

    /*cleanup*/
    free_real_matrix(A, n);
    free_real_matrix(B, n);
    free_real_matrix(C, n);

    return 0;
}
