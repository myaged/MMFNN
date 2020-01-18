#ifndef MATRIXOPS_H
#define MATRIXOPS_H

/*
    Matrix operations utility

    Part of MMFNN guiding code. Provided as is.
    Tested with C++11 (g++ ver. 4.9.4)
 */

#include <iostream>
#include <random>

using namespace std;

class MatrixOps {

    public:

        // -------------------------------------
        // gaussian random matrix builder
        // -------------------------------------
        static double** gaussianMatrixBuilder( double mu,
                                               double sigma,
                                               unsigned int numRows,
                                               unsigned int numCols ){

            double** matrix = new double*[numRows];
            for(unsigned int i=0; i<numRows; i++){
                matrix[i] = new double[numCols];
            }

            random_device rd{};
            mt19937 generator{rd()}; // using Mersenne twister
            normal_distribution<double> distribution(mu,sigma);

            for(unsigned int i=0; i<numRows; i++){
                for(unsigned int j=0; j<numCols; j++){
                    matrix[i][j] = distribution(generator);
                }
            }
            return matrix;
        }

        // -------------------------------------
        // dot product of two vectors
        // -------------------------------------
        static inline double dot(double* const &x, double* const &y, unsigned int const &numLatentFactors){
            double dotProduct = 0.0;
            for(unsigned int f=0; f<numLatentFactors; f++){
                dotProduct += x[f]*y[f];
            }
            return dotProduct;
        }

        // -------------------------------------
        // difference of dot products
        // -------------------------------------
        static inline double diffDot(double* const &x, double* const &y, double* const &z, unsigned int const &numLatentFactors){
            double diffDotProduct = 0.0;
            for(unsigned int f=0; f<numLatentFactors; f++){
                diffDotProduct += x[f]*(y[f]-z[f]);
            }
            return diffDotProduct;
        }

};

#endif
