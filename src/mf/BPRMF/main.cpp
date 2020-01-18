/*
    Example driver code
    - Performs training
    - Writes P, Q, and I_u^+ to files

    To compile : g++-4.9 -O3 -std=c++11 *.cpp -fopenmp -o main.x

    Part of MMFNN guiding code. Provided as is.
    Tested with C++11 (g++ ver. 4.9.4)
 */

#include <iostream>
#include "MatrixOps.h"
#include "Tuple.h"
#include "PBPR.h"
#include <unordered_map>
#include <unordered_set>
#include <fstream>
#include <sstream>

using namespace std;

int main(){

     // ------------------------------------
    // User input parameters
    // ------------------------------------

    // Train file must contain user indices 0,1,...,numUsers-1 and item indices 0,1,...,numItems-1
    string trainFile = "../../../data/ml1m/train.csv"; 
    
    unsigned int numUsers = 6040;
    unsigned int numItems = 3952;
    char fileDelimiter = '\t';
    constexpr int userItemRelevanceIndexes[3] = {0,1,2}; // {u,i,r]
    bool skipHeaderLine = false;

    // Output files
    string factorPFile = "output/ml1m/factorP.csv";
    string factorQFile = "output/ml1m/factorQ.csv";
    string userHistoryFile = "output/ml1m/userHistory.csv";

    // BPR parameters
    unsigned int numLatentFactors = 40;
    double mu = 0.0;
    double sigma = 0.01;
    double lambP = 0.0025;
    double lambQPlus = 0.0025;
    double lambQMinus = 0.00025;
    double eta = 0.01;
    unsigned int numEpochs = 64;
    unsigned int numCores = 4; // Choose 1 <= numCores <= Number of available cores

    // ------------------------------------
    // Read data
    // ------------------------------------
    cout << "reading training set ..." << endl;

    vector<Tuple> trainData; // list of (u,i,r)s
    ifstream dataStream;
    string line, field;
    int n = -1, ff;
    unsigned int user, item, relevance = 1;
    dataStream.open(trainFile);
    while (getline(dataStream,line)){
        n++;
        if (skipHeaderLine && n==0)    continue;
        istringstream lineStream(line);
        ff = 0;
        while (getline(lineStream, field, fileDelimiter)){
            switch(ff){
                case userItemRelevanceIndexes[0]:
                    user = stoi(field);
                    break;
                case userItemRelevanceIndexes[1]:
                    item = stoi(field);
                    break;
                case userItemRelevanceIndexes[2]:
                    relevance = stoi(field);
                    break;
            }
            ff++;
        }
        trainData.push_back({user,item,relevance});
    }
    dataStream.close();

    // ------------------------------------
    // Train
    // ------------------------------------
    cout << "initializing and learning model ..." << endl;
    
    PBPR pbpr(numUsers, numItems, numLatentFactors, mu, sigma, lambP, lambQPlus, lambQMinus, eta, numEpochs);

    struct timespec start, finish;
    double elapsed;
    clock_gettime(CLOCK_MONOTONIC, &start);

    pbpr.learn(trainData, numItems-1, numCores);

    // end elapsed time
    clock_gettime(CLOCK_MONOTONIC, &finish);
    elapsed = (finish.tv_sec - start.tv_sec);
    elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    cout << "*** Training - elapsed time : " << elapsed << " sec ***" << endl;

    trainData.clear();

    // ------------------------------------
    // Write to files
    // ------------------------------------

    ofstream outFile;

    // P
    cout << "Writing P to file ..." << endl;
    double** P = pbpr.getP();
    outFile.open(factorPFile);
    for(int i=0;i<numUsers;i++){;
        for(int j=0;j<numLatentFactors-1;j++){
            outFile << P[i][j] << ",";
        }
        outFile << P[i][numLatentFactors-1] << '\n';
    }
    outFile.close();

    // Q
    cout << "Writing Q to file ..." << endl;
    double** Q = pbpr.getQ();
    outFile.open(factorQFile);
    for(int i=0;i<numItems;i++){;
        for(int j=0;j<numLatentFactors-1;j++){
            outFile << Q[i][j] << ",";
        }
        outFile << Q[i][numLatentFactors-1] << '\n';
    }
    outFile.close();

    // I_u^+
    cout << "Writing user histories to file ..." << endl;
    unordered_map<unsigned int, unordered_set<unsigned int>> IPlus = pbpr.getIPlus();
    unordered_set<unsigned int> items;
    outFile.open(userHistoryFile);
    for (auto& kv : IPlus) {
        outFile << kv.first << '\t';
        items = kv.second;
        unsigned int ll = 0, lenItems = items.size();

        for (unsigned int item : items){
            ll++;
            outFile << item;
            if(ll <= lenItems-1) outFile << ',';
        }
        outFile << '\n';
    }
    outFile.close();

    return 0;
}
