#include <mpi.h>
#include <omp.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

int N, NX, NY;
double *X, *Y;
int left, right, up, down;
MPI_Request recvReq[4] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL};
MPI_Request sendReq[4] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL};
MPI_Status sendStat[4];
MPI_Status recvStat[4];
MPI_Datatype Vector;

#define c(i,j) ((j)*NX+(i))
#define hx(i)  (X[i+1]-X[i])
#define hy(j)  (Y[j+1]-Y[j])
#define laplasian(P,i,j)\
((-(P[NX*(j)+i+1]-P[NX*(j)+i])/hx(i)+(P[NX*(j)+i]-P[NX*(j)+i-1])/hx(i-1))/(0.5*(hx(i)+hx(i-1)))+\
(-(P[NX*(j+1)+i]-P[NX*(j)+i])/hy(j)+(P[NX*(j)+i]-P[NX*(j-1)+i])/hy(j-1))/(0.5*(hy(j)+hy(j-1))))
#define square(x) ((x)*(x))
#define EPS 0.0001

int
intLog2(int value) {
	// in task2 value is always 2^k
    int power = 0;
	while (value >>= 1) ++power;
	return power;
}

double
gridNode(int i) {
    return 2.0 * i / N;
}

double
Phi(double x, double y) {
    return exp(1 - square(x + y));
}

void
FillF(double *F) {
    int i, j;
    #pragma omp parallel for
    for(j = 1; j < NY - 1; ++j)
        for(i = 1; i < NX - 1; ++i)
            F[c(i,j)] = 4 * (1 - 2 * square(X[i] + Y[j])) * exp(1 - square(X[i] + Y[j]));
}

double
Product(double *A, double *B) {   
    int i, j;
    double local, global;

    local = 0.0;
    #pragma omp parallel for reduction(+:local)
    for(j = 1; j < NY - 1; ++j)
        for(i = 1; i < NX - 1; ++i)
            local += A[c(i,j)] * B[c(i,j)] * 0.25 * (hx(i) + hx(i-1)) * (hy(j) + hy(j-1));
    MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return global;
}

void
Receive(double *A) {
    if (left >= 0)
        MPI_Irecv(&A[c(0,0)], 1, Vector, left, 1, MPI_COMM_WORLD, &recvReq[0]);
    if (right >= 0)
        MPI_Irecv(&A[c(NX-1,0)], 1, Vector, right, 2, MPI_COMM_WORLD, &recvReq[1]);
    if (up >= 0)
        MPI_Irecv(&A[c(0,0)], NX, MPI_DOUBLE, up, 3, MPI_COMM_WORLD, &recvReq[2]);
    if (down >= 0)
        MPI_Irecv(&A[c(0,NY-1)], NX, MPI_DOUBLE, down, 4, MPI_COMM_WORLD, &recvReq[3]);
}

void
Send(double *A) {
    if (left >= 0)
        MPI_Isend(&A[c(1,0)], 1, Vector, left, 2, MPI_COMM_WORLD, &sendReq[0]);
    if (right >= 0)
        MPI_Isend(&A[c(NX-2,0)], 1, Vector, right, 1, MPI_COMM_WORLD, &sendReq[1]);
    if (up >= 0)
        MPI_Isend(&A[c(0,1)], NX, MPI_DOUBLE, up, 4, MPI_COMM_WORLD, &sendReq[2]);
    if (down >= 0)
        MPI_Isend(&A[c(0,NY-2)], NX, MPI_DOUBLE, down, 3, MPI_COMM_WORLD, &sendReq[3]);
}

void
Wait() {
    MPI_Waitall(4, sendReq, sendStat);
    MPI_Waitall(4, recvReq, recvStat);
}

void
FillR(double *R, double *P, double *F) {
    Receive(R);
    int i, j;
    #pragma omp parallel for
    for(j = 1; j < NY - 1; ++j)
        for(i = 1; i < NX - 1; ++i)
            R[c(i,j)] = laplasian(P,i,j) - F[c(i,j)];
    Send(R);
    Wait();
}

void
Update(double *A, double *B, double t, double* C) {
    Receive(A);
    int i, j;
    #pragma omp parallel for
    for(j = 1; j < NY - 1; ++j)
        for(i = 1; i < NX - 1; ++i)
            A[c(i,j)] = B[c(i,j)] - t * C[c(i,j)];
    Send(A);
    Wait();
}

void
FillLaplasian(double *A, double *laplA) {
    int i, j;
    #pragma omp parallel for
    for(j = 1; j < NY - 1; ++j)
        for(i = 1; i < NX - 1; ++i)
            laplA[c(i,j)] = laplasian(A,i,j);
}

double
MaxNorm(double *A, double t) {
	double maxD = 0.0, globalMax;
    int i, j;
    #pragma omp parallel for reduction(max:maxD)
    for(j = 1; j < NY - 1; ++j)
        for(i = 1; i < NX - 1; ++i)
			maxD = maxD > fabs(A[c(i,j)] * t) ? maxD : fabs(A[c(i,j)] * t);
    MPI_Allreduce(&maxD, &globalMax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
	return globalMax;
}

int
main(int argc, char **argv) {
    int i, j, iter = 1, proc;
    int numProc, rank, dimProc[2], coords[2];
    int sumPower, dimPower[2];
    int block[2], extra[2], begin[2];
    int periods[2] = {0,0};
    double *F, *P, *R, *G, *laplR, *laplG, *GT;
    double dot, laplDot, t, a, curE;
    MPI_Comm Grid_Comm;
    double *result;
    double time_beg, time_end;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numProc);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    MPI_Barrier(MPI_COMM_WORLD);
    time_beg = MPI_Wtime();

    N = atoi(argv[1]);
    
    sumPower = intLog2(numProc);
    dimPower[0] = sumPower >> 1;
    dimPower[1] = sumPower - dimPower[0];

    for(i = 0; i < 2; ++i) {
		// number of procs for dimension
        dimProc[i] = (unsigned int)1 << dimPower[i];
	}
	
	// create rectangle grid
    MPI_Cart_create(MPI_COMM_WORLD, 2, dimProc, periods, 0, &Grid_Comm);
	// get rank on new comm
    MPI_Comm_rank(Grid_Comm, &rank);
	// get coords
    MPI_Cart_coords(Grid_Comm, rank, 2, coords);
	// get neighbours
    MPI_Cart_shift(Grid_Comm, 0, 1, &left, &right);
    MPI_Cart_shift(Grid_Comm, 1, 1, &up, &down);

    for(i = 0; i < 2; ++i) {
		// base block size
        block[i] = (N - 1) >> dimPower[i];
		// undistributed grid elems
        extra[i] = (N - 1) - block[i] * dimProc[i];
		// if blocks are filled then offset is blockLength*coord
        begin[i] = block[i] * coords[i] + extra[i];
		// store border for communiication with neighbours
        block[i] += 2;
		// now distribute extra elems
        if (coords[i] < extra[i]) {
			// add one to block
            ++block[i];
			// fix offset by removing all distributed extras
            begin[i] -= extra[i] - coords[i];
        }
    }

    NX = block[0];
    NY = block[1];
	// for communication
    MPI_Type_vector(NY, 1, NX, MPI_DOUBLE, &Vector);
    MPI_Type_commit(&Vector);

	/*** init grid coords ****/
    X = (double*) calloc(NX, sizeof(double));
    #pragma omp parallel for
    for(i = 0; i < NX; ++i)
        X[i] = gridNode(begin[0] + i);

    Y = (double*) calloc(NY, sizeof(double));
    #pragma omp parallel for
    for(j = 0; j < NY; ++j)
        Y[j] = gridNode(begin[1] + j);
	/*************************/

	// compute part of F
    F = (double*) calloc(NX * NY, sizeof(double));
    FillF(F);

	// compute part of Phi
    GT = (double*) calloc(NX * NY, sizeof(double));
    #pragma omp parallel for
    for(j = 0; j < NY; ++j)
        for(i = 0; i < NX; ++i)
            GT[c(i,j)] = Phi(X[i], Y[j]);

	// if neighbours are undefined -> fill correspondent border with Phi
    P = (double*) calloc(NX * NY, sizeof(double));
    #pragma omp parallel for
    for(i = 0; i < NX; ++i) {
        if(up < 0)
            P[c(i,0)] = Phi(X[i], Y[0]);
        if(down < 0)
            P[c(i,NY-1)] = Phi(X[i], Y[NY-1]);
    }
    #pragma omp parallel for
    for(j = 0; j < NY; ++j) {
        if(left < 0)
            P[c(0,j)] = Phi(X[0], Y[j]);
        if(right < 0)
            P[c(NX-1,j)] = Phi(X[NX-1], Y[j]);
    }

	// compute 1st iteration residual
    R = (double*) calloc(NX * NY, sizeof(double));
    FillR(R, P, F);

	// compute 1st iteration tau
    laplR = (double*) calloc(NX * NY, sizeof(double));
    FillLaplasian(R, laplR);
    dot = Product(R, R);
    laplDot = Product(laplR, R);
    t = dot / laplDot;
	
	// next iteration
    Update(P, P, t, R);

    G = R;
    laplG = laplR;
    R = (double*) calloc(NX * NY, sizeof(double));
    laplR = (double*) calloc(NX * NY, sizeof(double));

    curE = MaxNorm(G, t);

	// compute iterations untill stop-criterion
    while(curE >= EPS) {
        iter++;
        FillR(R, P, F);
        FillLaplasian(R, laplR);
        a = Product(laplR, G) / laplDot;
        Update(G, R, a, G);
        FillLaplasian(G, laplG);
        laplDot = Product(laplG, G);
        t = Product(R, G) / laplDot;
        Update(P, P, t, G);
        curE = MaxNorm(G, t);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    time_end = MPI_Wtime();

	// compute error
    Update(GT, P, 1, GT);
    curE = MaxNorm(GT, 1.0);

    if(rank == 0) {
        printf("%d %d %d %lf %lf\n", numProc, N, iter, time_end - time_beg, curE);
        result = (double*) calloc((N + 1) * (N + 1), sizeof(double));
        for(proc = 0; proc < numProc; ++proc) {
            if(proc > 0) {
                MPI_Recv(block, 2, MPI_INT, proc, 5, MPI_COMM_WORLD, &recvStat[0]);
                NX = block[0];
                NY = block[1];
                MPI_Recv(begin, 2, MPI_INT, proc, 6, MPI_COMM_WORLD, &recvStat[1]);
                MPI_Recv(P, NX * NY, MPI_DOUBLE, proc, 7, MPI_COMM_WORLD, &recvStat[2]);
            }
            for(j = 0; j < NY; ++j)
                for(i = 0; i < NX; ++i)
                    result[(begin[1] + j) * (N + 1) + begin[0] + i] = P[c(i,j)];
        }
        
        for(j = 0; j < N + 1; ++j) {
            for(i = 0; i < N + 1; ++i)
                printf("%lf ", result[j * (N + 1) + i]);
            printf("\n");
        }
    } else {
        MPI_Send(block, 2, MPI_INT, 0, 5, MPI_COMM_WORLD);
        MPI_Send(begin, 2, MPI_INT, 0, 6, MPI_COMM_WORLD);        
        MPI_Send(P, NX * NY, MPI_DOUBLE, 0, 7, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}
