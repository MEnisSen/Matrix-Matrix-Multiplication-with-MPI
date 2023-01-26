/*
		
		Muhammed Enis Şen

		The following code calculates the wall clock time of matrix mutliplication(both sequential
		and parallel) of square matrices with sizes ranging from 5000 to 25000 with increments of 
		5000 with respect to kij index ordering as it was among the best in the previous assignment.
		The calculated times will be printed into 'result_times_serial.txt' and 'result_times_parallel.txt'
		file within the same directory.

		To compile selected indexings and run on a range of matrix sizes use the following
		commands:

		mpicc hw2.c -Dserial -Dparallel
		mpirun -np 1 ./a.out 1 5

		For varying core counts, simply add "-DparallelCore" and add two numbers on the execution line
		(Change the matrix sizes at 173 and 174 to perform the multiplication for desired sizes):

		mpicc hw2.c -DparallelCore -Dparallel
		mpirun -np 4 ./a.out 1 1

*/

#include "mpi.h"
#include "stdio.h"
#include "stdlib.h"
#include <time.h>
#include <unistd.h>
#include <sys/time.h>

double randBtw(double lb, double ub){
	double range = ub - lb;
	double div = RAND_MAX / range;
	
	return lb + (rand()/div);
}

void fill2dMatRand(double **X, int rows, int cols){
	int i, j;
	for(i=0;i<rows;i++)
		for(j=0;j<cols;j++)
			X[i][j] = randBtw(0., 1000.);
}

void fill1dMatRand(double *X, int rows, int cols){
	int i, j;
	for(i=0;i<rows;i++)
		for(j=0;j<cols;j++)
			X[i*rows+j] = randBtw(0., 1000.);
}

void matMatMultkij(double **A, double **B, double **C, int row1, int col1, int col2){
	int i,j,k;
	for(k=0;k<col1;k++){
		for(i=0;i<row1;i++){
			for(j=0;j<col2;j++)
				C[i][j] += A[i][k] * B[k][j];
		}
	}
}

void matMat1dMultkij(double *A, double *B, double *C, int row1, int col1, int col2){
	int i,j,k;
	for(k=0;k<col1;k++){
		for(i=0;i<row1;i++){
			for(j=0;j<col2;j++)
				C[i*row1+j] += A[i*row1+k] * B[k*col1+j];
		}
	}
}

void check(double *X, int mat_total_size, int rank){
	int i,cnt=0;
	for(i=0;i<mat_total_size;i++)
		if(X[i]==0.)
			cnt++;
	//printf("Got %d 0's in %d! rank - %d\n", cnt, mat_total_size, rank);
}

int main(int argc, char *argv[]){

    int size, rank;

    // Get and prepare the matrix sizes according to given argument n
	int matrix_size_lb = atoi(argv[1]), matrix_size_ub = atoi(argv[2]);
	int matrix_sizes = matrix_size_ub - matrix_size_lb + 1;
	int mat_n[matrix_sizes], i, m, j, r;
	for(i=0;i<matrix_sizes;i++){
		mat_n[i] = 5000 * (matrix_size_lb + i);
	}

	// File
	FILE *fptr;

	#ifdef serial // SERIAL --------------------------------------------------
	
	MPI_Status status;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	srand(time(NULL));

	// Perform it on only 1 core
	if(rank == 0){
		// Apply the same operations on each size
		for(m=0;m<matrix_sizes;m++){

			// Matrix size
			int mat_size = mat_n[m];

			// Allocate memory for matrices A, B and the result matrix C
			double **A, **B, **C;
			A = (double **)malloc(mat_size * sizeof(double *));
			for(i=0;i<mat_size;i++){
				A[i] = (double *)malloc(mat_size * sizeof(double));
			}
			B = (double **)malloc(mat_size * sizeof(double *));
			for(i=0;i<mat_size;i++){
				B[i] = (double *)malloc(mat_size * sizeof(double));
			}
			C = (double **)malloc(mat_size * sizeof(double *));
			for(i=0;i<mat_size;i++){
				C[i] = (double *)malloc(mat_size * sizeof(double));
			}

			// Fill A and B with random values between 0 and 1000
			fill2dMatRand(A, mat_size, mat_size);
			fill2dMatRand(B, mat_size, mat_size);

			// Time before multiplication operation
			struct timeval t;
			double time1, time2, duration;
			gettimeofday(&t, NULL);
			time1 = t.tv_sec + 1.0e-6*t.tv_usec;

			// Multiplication operation
			matMatMultkij(A, B, C, mat_size, mat_size, mat_size);

			// Time after operation and time difference
			gettimeofday(&t, NULL);
			time2 = t.tv_sec + 1.0e-6*t.tv_usec;

			// Open file
			fptr = fopen("result_times_serial.txt", "a+");
			// Print resulting time in the according txt file
			fprintf(fptr, "%d %d %lf\n", 1, mat_size, time2 - time1);
			// Close file
			fclose(fptr);

			// Free allocated memory before next operation with higher sizes
			for(i=0;i<mat_size;i++){
				free(A[i]);
				free(C[i]);
			}
			free(A);
			free(C);
			for(i=0;i<mat_size;i++){
				free(B[i]);
			}
			free(B);
		}
	}

	MPI_Finalize();

	#endif     // ------------------------------------------------------------

	#ifdef parallelCore // PARALLEL FOR VARIUOS CORE COUNTS ------------------

	// For varying core sizes, add this ifdef part in terminal
	mat_n[0]=5000; mat_n[1]=15000; mat_n[2]=25000;
	matrix_sizes=3;

	#endif     // ------------------------------------------------------------
	
    	#ifdef parallel // PARALLEL ----------------------------------------------
	
	double mult_start, mult_stop, comm_start, comm_stop;

	MPI_Status status;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	// Unique random values on each core
	srand(time(NULL) + rank);

	// Apply the same operations on each size
	for(m=0;m<matrix_sizes;m++){

		// Matrix size
		int mat_size = mat_n[m];

		// Calculate how many rows each core will get
		int worker_rows = mat_size / (size - 1);
		int master_rows = mat_size - worker_rows * (size - 1);
		// A unique variable for each core that takes different values if master or worker
		int core_mat_size = (rank != 0) ? worker_rows : master_rows;

		// Matrix memory allocations
		double *A_scat = malloc(mat_size * mat_size * sizeof(double));
		double *B = malloc(mat_size * mat_size * sizeof(double));
	    double *RES = malloc(mat_size * mat_size * sizeof(double));
		double *A = malloc(core_mat_size * mat_size * sizeof(double));
		double *C = malloc(core_mat_size * mat_size * sizeof(double));

		// Fill the initial A and B matrices in the master core
		if(rank == 0){
			fill1dMatRand(A_scat, mat_size, mat_size);
			fill1dMatRand(B, mat_size, mat_size);
		}
		
		// Prepare receive_cnt and receive_disp for MPI_Scatterv and MPI_Gatherv
		int *receive_cnt = malloc(size * sizeof(int));
		for(r = 0; r<size ; r++)
			receive_cnt[r] = (r != 0) ? worker_rows * mat_size : master_rows * mat_size;

		int *receive_disp = malloc(size * sizeof(int));
		receive_disp[0] = 0;
		receive_disp[1] = master_rows * mat_size;
		for(r = 2; r<size ; r++)
			receive_disp[r] = receive_disp[r-1] + worker_rows * mat_size;

		/*if(rank==0){
			for(int r=0;r<size;r++)
				printf("%d ", receive_cnt[r]);
			printf("\n");
			for(int r=0;r<size;r++)
				printf("%d ", receive_disp[r]);
			printf("\n");
		}*/

		// Scatter the initial A matrix
		MPI_Scatterv(A_scat, receive_cnt, receive_disp, MPI_DOUBLE,
					 A, core_mat_size*mat_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		// Broadcast B to all cores
		MPI_Bcast(B, mat_size * mat_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

		// Check if received matrix's size is as expected
		//check(A, core_mat_size*mat_size, rank);

		// Multiplication operation with all cores starting at the same time
		MPI_Barrier(MPI_COMM_WORLD);
		mult_start = MPI_Wtime();
		matMat1dMultkij(A, B, C, core_mat_size, mat_size, mat_size);
		mult_stop = MPI_Wtime();

		// Check if calculated matrix's size is as expected
		//check(C, core_mat_size*mat_size, rank);

		// Gather the calculated results into the master core
		MPI_Barrier(MPI_COMM_WORLD);
		comm_start = MPI_Wtime();
		MPI_Gatherv(C, core_mat_size * mat_size, MPI_DOUBLE, 
					RES, receive_cnt, receive_disp, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		comm_stop = MPI_Wtime();

		// Calculate total time with respect to master core
		double total_time = (mult_stop - mult_start) + (comm_stop - comm_start);

		/*if(rank==0)
			printf("cores-%d, size-%d, time-%lf\n", size, mat_size, total_time);*/


		if (rank == 0){
			// Open file
			fptr = fopen("result_times_parallel.txt", "a+");
			// Print resulting time in the according txt file
			fprintf(fptr, "%d %d %lf\n", size, mat_size, total_time);
			// Close file
			fclose(fptr);
		}
		
		// Free allocated memory before next operation with higher sizes
		free(A);
		free(A_scat);
		free(C);
		free(B);
		free(RES);
		free(receive_disp);
		free(receive_cnt);
		
		// Make sure cores start the next size at the same time
		MPI_Barrier(MPI_COMM_WORLD);
	}

	MPI_Finalize();

	#endif     // ------------------------------------------------------------


    return 0;
}
