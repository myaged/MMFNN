#ifndef EP_H
#define EP_H

/*
    EP with and without min. heap

    Part of MMFNN guiding code. Provided as is.
    Tested with C++11 (g++ ver. 4.9.4)
 */

#include <iostream>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <queue>
#include "helper.h"

using namespace std;

class EP{

    private:

        unsigned int numUsers;
        unsigned int numItems;
        unsigned int numLatentFactors;

        double **factorQ;
        double **factorP;
        unordered_map<unsigned int, unordered_set<unsigned int>> mapUserHistory;

        unordered_set<unsigned int> currentHistoryItems;
        vector<ScorePair> vecScorePairs; // holds (item,score) pairs
        unsigned int *topNList; // holds top-N list for a user

        priority_queue<ScorePair> pq; // for min. heap

    public:

        // ---------------------------------
        // Constructor
        // ---------------------------------
        EP( unsigned int numUsers,
            unsigned int numItems,
            unsigned int numLatentFactors,
            double **factorQ,
            double **factorP,
            unordered_map<unsigned int, unordered_set<unsigned int>> mapUserHistory) {
            
            this->numUsers = numUsers;
            this->numItems = numItems;
            this->numLatentFactors = numLatentFactors;
            this->factorQ = factorQ;
            this->factorP = factorP;
            this->mapUserHistory = mapUserHistory;
            this->vecScorePairs.resize(numItems);
        }

        // ---------------------------------
        // top-N prediction without min. heap
        // ---------------------------------
        unsigned int* predictTopN(unsigned int user, unsigned int N){

            for(unsigned int i=0; i<this->numItems; i++){
                double score = 0.0;
                for(unsigned int f=0; f<this->numLatentFactors; f++){
                    score += this->factorP[user][f] * this->factorQ[i][f];
                }
                vecScorePairs[i].index = i;
                vecScorePairs[i].value = score;
            }

            sort(vecScorePairs.begin(), vecScorePairs.end());

            // get top-N
            currentHistoryItems = mapUserHistory[user];
            unsigned int n=0;
            topNList = new unsigned int[N]();
            for(unsigned int i=0; i<this->numItems; i++){
                if( n<N ){
                    // exclude items already in user history
                    if ( currentHistoryItems.find(vecScorePairs[i].index) ==
                            currentHistoryItems.end() ){
                        topNList[n] = vecScorePairs[i].index;
                        n++;
                    }
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

            for(unsigned int i=0; i<this->numItems; i++){
                // exclude items already in user history
                if ( currentHistoryItems.find(i) == currentHistoryItems.end() ){
                    double score = 0.0;
                    for(unsigned int f=0; f<this->numLatentFactors; f++){
                        score += this->factorP[user][f] * this->factorQ[i][f];
                    }

                    if (pq.size() == N){
                        if (pq.top().value < score) {
                            pq.pop();
                            pq.push({i,score});
                        }
                    } else {
                        pq.push({i,score});
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
