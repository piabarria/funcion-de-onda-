#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <immintrin.h>

/* parámetros */
#define NX 64
#define NY 64
#define NZ 64

#define DX 0.1
#define DY 0.1
#define DZ 0.1

#define DT 0.01
#define C 1.0

#define NSTEPS 100

/* indexación */
#define IDX(i,j,k) ((i)*NY*NZ + (j)*NZ + (k))

/* condiciones de borde */
void apply_boundary_conditions(double *u)
{
    for(int i=0;i<NX;i++)
        for(int j=0;j<NY;j++)
            for(int k=0;k<NZ;k++)
                if(i==0||i==NX-1||j==0||j==NY-1||k==0||k==NZ-1)
                    u[IDX(i,j,k)] = 0.0;
}

/* paso temporal con AVX */
void step(const double *u_prev,const double *u_curr,double *u_next)
{
    double c2dt2 = C*C*DT*DT;

    __m256d two = _mm256_set1_pd(2.0);
    __m256d c2 = _mm256_set1_pd(c2dt2);
    __m256d dx2 = _mm256_set1_pd(DX*DX);
    __m256d dy2 = _mm256_set1_pd(DY*DY);
    __m256d dz2 = _mm256_set1_pd(DZ*DZ);

    for(int i=1;i<NX-1;i++)
    {
        for(int j=1;j<NY-1;j++)
        {
            int k;

            for(k=1;k<NZ-1-3;k+=4)
            {
                int idx = IDX(i,j,k);

                __m256d u_c   = _mm256_loadu_pd(&u_curr[idx]);
                __m256d u_p   = _mm256_loadu_pd(&u_prev[idx]);

                __m256d u_ip1 = _mm256_loadu_pd(&u_curr[IDX(i+1,j,k)]);
                __m256d u_im1 = _mm256_loadu_pd(&u_curr[IDX(i-1,j,k)]);

                __m256d u_jp1 = _mm256_loadu_pd(&u_curr[IDX(i,j+1,k)]);
                __m256d u_jm1 = _mm256_loadu_pd(&u_curr[IDX(i,j-1,k)]);

                __m256d u_kp1 = _mm256_loadu_pd(&u_curr[IDX(i,j,k+1)]);
                __m256d u_km1 = _mm256_loadu_pd(&u_curr[IDX(i,j,k-1)]);

                __m256d d2x = _mm256_div_pd(
                    _mm256_sub_pd(_mm256_add_pd(u_ip1,u_im1), _mm256_mul_pd(two,u_c)),
                    dx2
                );

                __m256d d2y = _mm256_div_pd(
                    _mm256_sub_pd(_mm256_add_pd(u_jp1,u_jm1), _mm256_mul_pd(two,u_c)),
                    dy2
                );

                __m256d d2z = _mm256_div_pd(
                    _mm256_sub_pd(_mm256_add_pd(u_kp1,u_km1), _mm256_mul_pd(two,u_c)),
                    dz2
                );

                __m256d lap = _mm256_add_pd(_mm256_add_pd(d2x,d2y), d2z);

                __m256d result = _mm256_add_pd(
                    _mm256_sub_pd(_mm256_mul_pd(two,u_c), u_p),
                    _mm256_mul_pd(c2, lap)
                );

                _mm256_storeu_pd(&u_next[idx], result);
            }

            /* resto escalar */
            for(;k<NZ-1;k++)
            {
                double d2x =
                (u_curr[IDX(i+1,j,k)]
                -2*u_curr[IDX(i,j,k)]
                +u_curr[IDX(i-1,j,k)])/(DX*DX);

                double d2y =
                (u_curr[IDX(i,j+1,k)]
                -2*u_curr[IDX(i,j,k)]
                +u_curr[IDX(i,j-1,k)])/(DY*DY);

                double d2z =
                (u_curr[IDX(i,j,k+1)]
                -2*u_curr[IDX(i,j,k)]
                +u_curr[IDX(i,j,k-1)])/(DZ*DZ);

                double lap = d2x + d2y + d2z;

                u_next[IDX(i,j,k)] =
                2*u_curr[IDX(i,j,k)]
                -u_prev[IDX(i,j,k)]
                +c2dt2*lap;
            }
        }
    }

    apply_boundary_conditions(u_next);
}

/* inicialización (pulso gaussiano) */
void initialize(double *u_prev,double *u_curr)
{
    for(int i=0;i<NX;i++)
    for(int j=0;j<NY;j++)
    for(int k=0;k<NZ;k++)
    {
        double x = i - NX/2;
        double y = j - NY/2;
        double z = k - NZ/2;

        double r2 = x*x + y*y + z*z;

        u_curr[IDX(i,j,k)] = exp(-r2/100.0);
        u_prev[IDX(i,j,k)] = u_curr[IDX(i,j,k)];
    }
}

void save_slice(double *u, int step)
{
    FILE *f;
    char filename[50];

    sprintf(filename, "output_%d.dat", step);
    f = fopen(filename, "w");

    int k = NZ/2;  // plano central en z

    for(int i=0;i<NX;i++)
    {
        for(int j=0;j<NY;j++)
        {
            fprintf(f,"%d %d %lf\n", i, j, u[IDX(i,j,k)]);
        }
        fprintf(f,"\n");
    }

    fclose(f);
}

/* main */
int main()
{
    int size = NX*NY*NZ;

    double *u_prev = malloc(size*sizeof(double));
    double *u_curr = malloc(size*sizeof(double));
    double *u_next = malloc(size*sizeof(double));

    initialize(u_prev,u_curr);

    for(int t=0;t<NSTEPS;t++)
{
    step(u_prev,u_curr,u_next);

    if(t % 10 == 0)  // guarda cada 10 pasos
        save_slice(u_curr, t);

    double *tmp = u_prev;
    u_prev = u_curr;
    u_curr = u_next;
    u_next = tmp;
}
    printf("Simulación terminada\n");

    free(u_prev);
    free(u_curr);
    free(u_next);

    return 0;
}