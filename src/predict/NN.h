#ifndef NN_H
#define NN_H

/*
    NN with and without min. heap

    Requires FLANN to be pre-installed. See:
    - https://github.com/mariusmuja/flann
    - http://www.cs.ubc.ca/research/flann

    Part of MMFNN guiding code. Provided as is.
    Tested with C++11 (g++ ver. 4.9.4) and flann-1.8.4
 */

#include <iostream>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <queue>
#include "helper.h"
#include <flann/flann.hpp>

using namespace std;

class NN{

    private:

        unsigned int numUsers;
        unsigned int numItems;
        unsigned int numLatentFactors;

        unsigned int K; // for knn
        flann::Matrix<int> knns;

        double **factorQ;
        double **factorP;
        unordered_map<unsigned int, unordered_set<unsigned int>> mapUserHistory;

        unordered_set<unsigned int> currentHistoryItems;
        vector<ScorePair> vecScorePairs; // holds (item,score) pairs
        unsigned int *topNList; // holds top-N list for a user

        priority_queue<ScorePair> pq; // for min. heap

        // ---------------------------------
        // flann converter
        // ---------------------------------
        flann::Matrix<double> Q2Flann() {

            size_t rows = this->numItems;
            size_t cols = this->numLatentFactors;
            size_t size = this->numItems*this->numLatentFactors;
            flann::Matrix<double>m(new double[size], rows, cols);

            for(size_t n = 0; n < size; ++n){
                *(m.ptr()+n) = this->factorQ[n/cols][n%cols];
            }

            return m;
        }


    public:

        // ---------------------------------
        // Constructor
        // ---------------------------------
        NN( unsigned int numUsers,
            unsigned int numItems,
            unsigned int numLatentFactors,
            unsigned int K,
            double **factorQ,
            double **factorP,
            unordered_map<unsigned int, unordered_set<unsigned int>> mapUserHistory) {

            this->numUsers = numUsers;
            this->numItems = numItems;
            this->numLatentFactors = numLatentFactors;
            this->K = K;
            this->factorQ = factorQ;
            this->factorP = factorP;
            this->mapUserHistory = mapUserHistory;
            this->vecScorePairs.resize(numItems);
        }

        // ---------------------------------
        // Build index and find knns
        // ---------------------------------
        void indexAndKnn( flann::flann_algorithm_t algorithm,
                          int kdtreeNumTrees,
                          int kmeansBranching,
                          int kmeansNumIterations,
                          int searchNumChecks,
                          int searchNumCores) {

            flann::log_verbosity(flann::FLANN_LOG_INFO);

            cout << "converting item factors to flann matrix ..." << endl;
            // This may be otherwise implemented in production implementations ...
            flann::Matrix<double> factorQFlann = Q2Flann();

            cout << "building index and finding knns ..." << endl;

            // read index params
            flann::IndexParams indexParameters;
            indexParameters["algorithm"] = algorithm;
            indexParameters["trees"] = kdtreeNumTrees;
            indexParameters["branching"] = kmeansBranching;
            indexParameters["iterations"] = kmeansNumIterations;

            // start elapsed time
            clock_gettime(CLOCK_MONOTONIC, &start);

            // build index
            flann::Index<flann::L2<double> > index(factorQFlann, indexParameters);
            index.buildIndex();

            // end elapsed time
            clock_gettime(CLOCK_MONOTONIC, &finish);
            elapsed = (finish.tv_sec - start.tv_sec);
            elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
            cout << "*** NN tree building - elapsed time :" << elapsed << " sec ***" << endl;

            // create flann matrices for knn
            flann::Matrix<int> knns(new int[factorQFlann.rows*(this->K+1)], factorQFlann.rows, this->K+1);
            flann::Matrix<double> knnDistances(new double[factorQFlann.rows*(this->K+1)], factorQFlann.rows, this->K+1);

            flann::SearchParams searchParameters = flann::SearchParams();
            searchParameters.checks = searchNumChecks;
            searchParameters.cores = searchNumCores;

            // start elapsed time
            clock_gettime(CLOCK_MONOTONIC, &start);

            // do a knn search
            cout << "doing a knn search ..." << endl;
            index.knnSearch(factorQFlann, knns, knnDistances, this->K+1, searchParameters);

            // end elapsed time
            clock_gettime(CLOCK_MONOTONIC, &finish);
            elapsed = (finish.tv_sec - start.tv_sec);
            elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
            cout << "*** NN finding for items - elapsed time :" << elapsed << " sec ***" << endl;

            delete[] factorQFlann.ptr();

            this->knns = knns;
        }

        // ---------------------------------
        // top-N prediction without min. heap
        // ---------------------------------
        unsigned int* predictTopN(unsigned int user, unsigned int N){

            currentHistoryItems = mapUserHistory[user];
            unordered_set<unsigned int> unionNeighbors;

            for (unsigned int historyItem : currentHistoryItems){
                for(unsigned int k=0; k<this->K+1;k++){
                    if (currentHistoryItems.find(this->knns[historyItem][k]) == currentHistoryItems.end() ){
                        // i.e. exclude items already in history
                        unionNeighbors.insert(this->knns[historyItem][k]);
                    }
                }
            }

            vector<ScorePair> vecScorePairs(unionNeighbors.size());
            unsigned int ii = 0;
            for (unsigned int neighbor : unionNeighbors){
                double score = 0.0;
                for(unsigned int f=0; f<this->numLatentFactors; f++){
                    score += factorP[user][f] * factorQ[neighbor][f];
                }
                vecScorePairs[ii].index = neighbor;
                vecScorePairs[ii].value = score;
                ii++;
            }

            sort(vecScorePairs.begin(), vecScorePairs.end());

            // get top-N
            unsigned int n=0;
            topNList = new unsigned int[N]();
            for(unsigned int i=0; i<vecScorePairs.size(); i++){
                if( n<N ){
                    topNList[n] = vecScorePairs[i].index;
                    n++;
                } else {
                    break;
                }
            }

            return topNList;
        }

        // ---------------------------------
        // top-N prediction using min. heap
        // ---------------------------------
        unsigned int* predictTopNWithMinHeap(unsigned int user, unsigned int N){

            currentHistoryItems = mapUserHistory[user];

            unordered_set<unsigned int> unionNeighbors;
            for (unsigned int historyItem : currentHistoryItems){
                for(unsigned int k=0; k<this->K+1;k++){
                    unsigned int neighbor = this->knns[historyItem][k];
                    if ( currentHistoryItems.find(neighbor) == currentHistoryItems.end() &&
                        unionNeighbors.find(neighbor) == unionNeighbors.end() ){
                        // i.e. exclude items already in history, and neighbors that are already handled

                        double score = 0.0;
                        for(unsigned int f=0; f<this->numLatentFactors; f++){
                            score += factorP[user][f] * factorQ[neighbor][f];
                        }
                        if (pq.size() == N){
                            if (pq.top().value < score) {
                                pq.pop();
                                pq.push({neighbor,score});
                            }
                        } else {
                            pq.push({neighbor,score});
                        }

                        unionNeighbors.insert(neighbor);
                    }
                }
            }

            // get top-N
            unsigned int n=N-1;
            topNList = new unsigned int[N]();
            while( !pq.empty() ) {
                topNList[n] = pq.top().index;
                pq.pop();
                n--;
            }

            return topNList;
        }

};

#endif
