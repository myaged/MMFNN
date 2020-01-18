/*
    Implementation of Parallel BPRMF based on PLtR-N
    (Yagci et al., On Parallelizing SGD for pairwise LtR in CF RSs, 2017)

    Part of MMFNN guiding code. Provided as is.
    Tested with C++11 (g++ ver. 4.9.4)
 */
 
#include "PBPR.h"

// -------------------------------------
// Update model
// -------------------------------------
void PBPR::updateParallel(){

    unsigned int lenData = data.size();

    random_device rd{};
    mt19937 generator{rd()};
    uniform_int_distribution<unsigned int> dataDistribution(0, lenData-1);
    random_device rd2{};
    mt19937 generator2{rd2()};
    uniform_int_distribution<unsigned int> itemDistribution(0, indexCounterItem-1);

    unsigned int epoch = 0;
    for( unsigned int k=0; k<this->numEpochs/this->numProcs; k++ ){
        cout << "epoch: " << epoch << endl;
        for( int j=0; j<lenData; j++ ){
            // sample with repetition
            unsigned int rnd = dataDistribution(generator);
            unsigned int user = data[rnd].getUserId();
            unsigned int posItem = data[rnd].getItemId();
            int negItem = -1;
            unsigned int numTrials = 0;
            while (numTrials < 10){
                unsigned int rnd2 = itemDistribution(generator2);
                if( IPlus[user].find(rnd2) == IPlus[user].end() ){
                    negItem = rnd2;
                    break;
                }
                numTrials += 1;
            }
            if( negItem != -1 ){

                double delta = 1.0 - this->sigmoid( MatrixOps::diffDot(P[user], Q[posItem], Q[negItem], this->numLatentFactors) );

                for(int f=0; f<this->numLatentFactors; f++){
                    P[user][f] += this->eta *
                    (delta * (Q[posItem][f] - Q[negItem][f]) - this->lambP * P[user][f]);
                }

                for(int f=0; f<this->numLatentFactors; f++){
                    Q[posItem][f] += this->eta *
                    (delta * P[user][f] - this->lambQPlus * Q[posItem][f]);
                }

                for(int f=0; f<this->numLatentFactors; f++){
                    Q[negItem][f] += this->eta *
                    (delta * -1.0*P[user][f] - this->lambQMinus * Q[negItem][f]);
                }

            }

        }
        epoch += 1;
    }
}

// -------------------------------------
// Sigmoid function
// -------------------------------------
double PBPR::sigmoid(double const &x){
    if( x > 0 ){
        return 1.0 / (1.0 + exp(-x));
    } else if (x <= 0) {
        double ex = exp(x);
        return ex / (1.0 + ex);
    } else {
        cout << "Sigmoid value error ..." << endl;
        return 0.0;
    }
}

// -------------------------------------
// Constructor
// -------------------------------------
PBPR::PBPR( int numUsers,
            int numItems,
            int numLatentFactors,
            double mu,
            double sigma,
            double lambP,
            double lambQPlus,
            double lambQMinus,
            double eta,
            int numEpochs) {

    this->numUsers = numUsers;
    this->numItems = numItems;
    this->numLatentFactors = numLatentFactors;
    this->P = MatrixOps::gaussianMatrixBuilder(mu, sigma, numUsers, numLatentFactors);
    this->Q = MatrixOps::gaussianMatrixBuilder(mu, sigma, numItems, numLatentFactors);
    this->lambP = lambP;
    this->lambQPlus = lambQPlus;
    this->lambQMinus = lambQMinus;
    this->eta = eta;
    this->numEpochs = numEpochs;
}

// -------------------------------------
// Learn model
// -------------------------------------
void PBPR::learn(vector<Tuple>& data, unsigned int indexCounterItem, unsigned int numProcs){

    // build user histories in the first pass
    for( Tuple& uir : data){
        unsigned int user = uir.getUserId();
        unsigned int item = uir.getItemId();

        if( IPlus.find(user) == IPlus.end() ){
            unordered_set<unsigned int> setHistoryItems;
            IPlus[user] = setHistoryItems;
        }
        IPlus[user].insert(item);
    }

    // parallel processing coordination
    this->data = data;
    this->indexCounterItem = indexCounterItem;
    this->numProcs = numProcs;
    omp_set_dynamic(0);
    omp_set_num_threads(this->numProcs);
    #pragma omp parallel
    {
        this->updateParallel();
    }

}

// -------------------------------------
// Getter for P
// -------------------------------------
double** PBPR::getP() const{
    return this->P;
}

// -------------------------------------
// Getter for Q
// -------------------------------------
double** PBPR::getQ() const{
    return this->Q;
}

// -------------------------------------
// Getter for IPlus
// -------------------------------------
unordered_map<unsigned int, unordered_set<unsigned int>> PBPR::getIPlus() const{
    return this->IPlus;
}

