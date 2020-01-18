#ifndef PBPR_H
#define PBPR_H

/*
    Interface of Parallel BPRMF based on PLtR-N
    (Yagci et al., On Parallelizing SGD for pairwise LtR in CF RSs, 2017)

    Part of MMFNN guiding code. Provided as is.
    Tested with C++11 (g++ ver. 4.9.4)
 */

#include <iostream>
#include <cmath>
#include <omp.h>
#include "MatrixOps.h"
#include "Tuple.h"
#include <unordered_map>
#include <unordered_set>

using namespace std;

class PBPR{

    private:

        unsigned int numUsers;
        unsigned int numItems;
        unsigned int numLatentFactors;
        double** P; // user component matrix
        double** Q; // item component matrix
        double lambP; // regularization parameter
        double lambQPlus; // regularization parameter
        double lambQMinus; // regularization parameter
        double eta; // learning rate
        unsigned int numEpochs; // number of training epochs
        unordered_map<unsigned int,unordered_set<unsigned int>> IPlus; // user histories

        // global vars for parallelization
        vector<Tuple> data;
        unsigned int indexCounterItem;
        unsigned int numProcs;

        void updateParallel();
        double sigmoid(double const &x);

    public:

        PBPR( int numUsers,
              int numItems,
              int numLatentFactors,
              double mu,
              double sigma,
              double lambP,
              double lambQPlus,
              double lambQMinus,
              double eta,
              int numEpochs );
        
        void learn(vector<Tuple>& data, unsigned int indexCounterItem, unsigned int numProcs);
        double** getP() const;
        double** getQ() const;
        unordered_map<unsigned int, unordered_set<unsigned int>> getIPlus() const;

};

#endif
