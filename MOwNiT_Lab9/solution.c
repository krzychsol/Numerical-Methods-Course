#include <stdio.h>
#include <stdlib.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_blas.h>
#include <time.h>

void generate_matrix(gsl_matrix *A, int n) {
    // Generowanie losowej macierzy A
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            gsl_matrix_set(A, i, j, rand() % 1000);
        }
    }
}

void generate_vector(gsl_vector *b, int n) {
    // Generowanie losowego wektora b
    for (int i = 0; i < n; i++) {
        gsl_vector_set(b, i, rand() % 1000);
    }
}

void print_matrix(gsl_matrix *A) {
    int n = A->size1;
    int m = A->size2;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            printf("%.2f ", gsl_matrix_get(A, i, j));
        }
        printf("\n");
    }
}

void print_vector(gsl_vector *b) {
    int n = b->size;
    for (int i = 0; i < n; i++) {
        printf("%.2f ", gsl_vector_get(b, i));
    }
    printf("\n");
}

void solve_lu(gsl_matrix *A, gsl_vector *b, gsl_vector *x) {
    int n = A->size1;
    gsl_matrix *LU = gsl_matrix_alloc(n, n);
    gsl_matrix_memcpy(LU, A);

    gsl_permutation *p = gsl_permutation_alloc(n);
    int signum;

    gsl_linalg_LU_decomp(LU, p, &signum);
    gsl_linalg_LU_solve(LU, p, b, x);

    printf("Rozwiązanie układu równań (dekompozycja LU):\n");
    print_vector(x);

    gsl_matrix_free(LU);
    gsl_permutation_free(p);
}

void solve_inverse(gsl_matrix *A, gsl_vector *b, gsl_vector *x) {
    int n = A->size1;
    gsl_matrix *invA = gsl_matrix_alloc(n, n);
    gsl_matrix_memcpy(invA, A);

    gsl_permutation *p = gsl_permutation_alloc(n);
    int signum;

    gsl_linalg_LU_decomp(invA, p, &signum);
    gsl_linalg_LU_invert(invA, p, invA);

    gsl_blas_dgemv(CblasNoTrans, 1.0, invA, b, 0.0, x);

    printf("Rozwiązanie układu równań (odwrócenie macierzy):\n");
    print_vector(x);

    gsl_matrix_free(invA);
    gsl_permutation_free(p);
}

void solve_qr(gsl_matrix *A, gsl_vector *b, gsl_vector *x) {
    int n = A->size1;
    gsl_matrix *QR = gsl_matrix_alloc(n, n);
    gsl_matrix_memcpy(QR, A);

    gsl_vector *tau = gsl_vector_alloc(n);

    gsl_linalg_QR_decomp(QR, tau);
    gsl_linalg_QR_solve(QR, tau, b, x);

    printf("Rozwiązanie układu równań (dekompozycja QR):\n");
    print_vector(x);
    
    gsl_matrix_free(QR);
    gsl_vector_free(tau);
}

int check_solution(gsl_matrix *A, gsl_vector *b, gsl_vector *x) {
    gsl_vector *Ax = gsl_vector_alloc(b->size);
    gsl_blas_dgemv(CblasNoTrans, 1.0, A, x, 0.0, Ax);

    double epsilon = 1e-6; // Tolerancja

    int isEqual = 1; // Zmienna oznaczająca równość wektorów

    for (size_t i = 0; i < b->size; i++) {
        double diff = gsl_vector_get(Ax, i) - gsl_vector_get(b, i);
        if (fabs(diff) > epsilon) {
            isEqual = 0;
            break;
        }
    }

    gsl_vector_free(Ax);

    return isEqual;
}


void measure_time(clock_t start, const char *method_name) {
    clock_t end = clock();
    double cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;

    printf("Czas wykonania (%s): %.6f s\n", method_name, cpu_time_used);
}

int main(int argc, char *argv[]) {
    srand(time(NULL));

    if (argc < 2) {
        printf("Podaj rozmiar układu równań jako argument przy uruchomieniu programu.\n");
        return 1;
    }

    int n = atoi(argv[1]);

    gsl_matrix *A = gsl_matrix_alloc(n, n);
    gsl_vector *b = gsl_vector_alloc(n);

    generate_matrix(A, n);
    generate_vector(b, n);

    printf("Macierz układu A:\n");
    print_matrix(A);

    printf("Wektor wyrazów wolnych b:\n");
    print_vector(b);

    // Rozwiązanie przez dekompozycję LU
    gsl_vector *x_lu = gsl_vector_alloc(n);  // Wektor rozwiązania
    clock_t start = clock();
    solve_lu(A, b, x_lu);
    measure_time(start, "dekompozycja LU");
    
    // Zapisanie czasu wykonania do pliku
    FILE *lu_file = fopen("LU.txt", "a");
    if (lu_file == NULL) {
        printf("Błąd przy otwieraniu pliku.\n");
        return 1;
    }
    fprintf(lu_file, "%d %f\n", n, ((double) (clock() - start)) / CLOCKS_PER_SEC);
    fclose(lu_file);

    // Sprawdzenie poprawności rozwiązania
    int isCorrect = check_solution(A, b, x_lu);
    printf("Rozwiązanie poprawne: %s\n", isCorrect ? "Tak" : "Nie");
    gsl_vector_free(x_lu);

    // Rozwiązanie przez dekompozycję QR
    gsl_vector *x_qr = gsl_vector_alloc(n);  // Wektor rozwiązania
    start = clock();
    solve_qr(A, b, x_qr);
    measure_time(start, "dekompozycja QR");

    // Zapisanie czasu wykonania do pliku
    FILE *qr_file = fopen("QR.txt", "a");
    if (qr_file == NULL) {
        printf("Błąd przy otwieraniu pliku.\n");
        return 1;
    }
    fprintf(qr_file, "%d %f\n", n, ((double) (clock() - start)) / CLOCKS_PER_SEC);
    fclose(qr_file);

    // Sprawdzenie poprawności rozwiązania
    isCorrect = check_solution(A, b, x_qr);
    printf("Rozwiązanie poprawne: %s\n", isCorrect ? "Tak" : "Nie");
    gsl_vector_free(x_qr);

    // Rozwiązanie przez odwrócenie macierzy
    gsl_vector *x_inv = gsl_vector_alloc(n);
    start = clock();
    solve_inverse(A, b, x_inv);
    measure_time(start, "odwracanie macierzy");

    // Zapisanie czasu wykonania do pliku
    FILE *inv_file = fopen("inv_matrix.txt", "a");
    if (inv_file == NULL) {
        printf("Błąd przy otwieraniu pliku.\n");
        return 1;
    }
    fprintf(inv_file, "%d %f\n", n, ((double) (clock() - start)) / CLOCKS_PER_SEC);
    fclose(inv_file);

    // Sprawdzenie poprawności rozwiązania
    isCorrect = check_solution(A, b, x_inv);
    printf("Rozwiązanie poprawne: %s\n", isCorrect ? "Tak" : "Nie");
    gsl_vector_free(x_inv);

    gsl_matrix_free(A);
    gsl_vector_free(b);

    return 0;
}

