/*
    Tester for prediction with MMFNN

    Requires FLANN to be pre-installed. See:
    - https://github.com/mariusmuja/flann
    - http://www.cs.ubc.ca/research/flann

    To compile : g++-4.9 -O3 -std=c++11 -I $FLANN_ROOT/include  main_NN.cpp -fopenmp -o main_NN.x

    Part of MMFNN guiding code. Provided as is.
    Tested with C++11 (g++ ver. 4.9.4) and flann-1.8.4

*/

# include <iostream>
#include "helper.h"
#include "NN.h"

using namespace std;

int main(){

    // ---------------------------------
    // Input parameters
    // ---------------------------------

    // factor and history files
    string factorQFile = "../mf/BPRMF/output/ml1m/factorQ.csv";
    string factorPFile = "../mf/BPRMF/output/ml1m/factorP.csv";
    string userHistoryFile = "../mf/BPRMF/output/ml1m/userHistory.csv";
    string testFile = "../../data/ml1m/test.csv";

    unsigned int numUsers = 6040;
    unsigned int numItems = 3952;
    unsigned int numLatentFactors = 40;

    // nn index params
    flann::flann_algorithm_t algorithm = flann::FLANN_INDEX_KDTREE; // KDTREE or KMEANS
    int kdtreeNumTrees = 8;
    int kmeansBranching = 32;
    int kmeansNumIterations = 5;

    // nn search params
    int K = 10; // for K nearest neighbors
    int searchNumChecks = 128;
    int searchNumCores = 2; // use 0 for all cores

    // for top-N
    unsigned int N = 10;
    unsigned int reportEvery = 1000;

    // ---------------------------------
    // Reading data
    // ---------------------------------
    cout << "reading item factors ..." << endl;
    double **factorQ = getFactors(factorQFile, numItems, numLatentFactors);

    cout << "reading user factors ..." << endl;
    double **factorP = getFactors(factorPFile, numUsers, numLatentFactors);

    cout << "reading user histories ..." << endl;
    unordered_map<unsigned int, unordered_set<unsigned int>> mapUserHistory =
        getUserHistory(userHistoryFile);

    cout << "reading test data ..." << endl;
    int userIndex = 0, itemIndex = 1;
    char delimiter = '\t';
    vector<UIPair> vecTestPairs = getTestData(testFile, userIndex, itemIndex, delimiter);

    // ---------------------------------
    // Find nearest neighbors
    // ---------------------------------

    NN nn(numUsers, numItems, numLatentFactors, K, factorQ, factorP, mapUserHistory);

    nn.indexAndKnn( algorithm,
                    kdtreeNumTrees, kmeansBranching, kmeansNumIterations,
                    searchNumChecks, searchNumCores);

    // ---------------------------------
    // top-N Predictions
    // ---------------------------------
    cout << "predicting ..." << endl;
    unsigned int hits = 0;
    unsigned int numRecs = 0;
    double mrr = 0.0;

    // start elapsed time
    clock_gettime(CLOCK_MONOTONIC, &start);

    // start evaluation
    unsigned int iterCount = 0;
    unsigned int *topNList; // holds top-N list for a user
    for(UIPair& lp : vecTestPairs){

        auto search = mapUserHistory.find(lp.user);
        if( search != mapUserHistory.end() ){

            topNList = nn.predictTopNWithMinHeap(lp.user, N);

            // *** This section can be commented if measuring execution time
            for(unsigned int n=0; n<N; n++){
                if(lp.item == topNList[n]){
                    hits++;
                    mrr += 1.0/(n+1);
                }
            }
            numRecs++;
            // *** End of section

        }
        if(iterCount%reportEvery == 0){
            cout << "tested: " << iterCount << endl;
        }
        iterCount++;
    }

    // end elapsed time
    clock_gettime(CLOCK_MONOTONIC, &finish);
    elapsed = (finish.tv_sec - start.tv_sec);
    elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    cout << "*** top-N prediction - elapsed time : " << elapsed << " sec ***" << endl;

    // communicate some more results
    cout << "num recs = " << numRecs << endl;
    cout << "hit rate = " << 1.*hits/numRecs << endl;
    cout << "mrr = " << mrr/numRecs << endl;

    return 0;
}
