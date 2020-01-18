#ifndef PREDICTIONS_HELPER_H
#define PREDICTIONS_HELPER_H

/*
    Common helper structs 
    + functions for file reading

    Part of MMFNN guiding code. Provided as is.
    Tested with C++11 (g++ ver. 4.9.4)
*/

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <unordered_map>
#include <unordered_set>

using namespace std;

// ---------------------------------
// Useful structs
// ---------------------------------

// (user,item) pairs
struct UIPair{
    unsigned int user;
    unsigned int item;
};

// (item,score) pairs
struct ScorePair{
    unsigned int index;
    double value;
    bool operator<(const ScorePair& rhs)const {
        return value > rhs.value; // achieves descending sort for vector, min. heap, ...
    }
};

// for measuring elapsed time
struct timespec start, finish;
double elapsed;

// ---------------------------------
// File reading stuff
// ---------------------------------

// a few variables useful for file reading
string line, field;
unsigned int ee, uu, ii, ff, columnNumber;

// reading factors
double** getFactors(string dataFactors, unsigned int numEntities, unsigned int numLatentFactors){
    double **factors = new double*[numEntities];
    ifstream ifs(dataFactors);
    if(ifs.is_open()){
        ee = 0;
        while(getline(ifs, line)){
            factors[ee] = new double[numLatentFactors];
            istringstream iss(line);
            ff = 0;
            while (getline(iss, field, ',')){
                factors[ee][ff] = stod(field);
                ff++;
            }
            ee++;
        }
    } else {
        cout << "ERROR: Unable to open file " << dataFactors << endl;
    }

    return factors;
}

// reading user histories
unordered_map<unsigned int, unordered_set<unsigned int>> getUserHistory(string dataUserHistory){
    unordered_map<unsigned int, unordered_set<unsigned int>> mapUserHistory;
    ifstream ifs(dataUserHistory);
    if(ifs.is_open()){
        while(getline(ifs, line)){
            istringstream iss(line);
            unordered_set<unsigned int> setHistoryItems;
            columnNumber = 0;
            while (getline(iss, field, '\t')){
                if(columnNumber == 0){
                    uu = stoi(field);
                } else {
                    stringstream ss(field);
                    while(ss.good()){
                        string historyItem;
                        getline(ss, historyItem, ',');
                        setHistoryItems.insert(stoi(historyItem));
                    }
                }
                columnNumber++;
            }
            mapUserHistory[uu] = setHistoryItems;
        }
    } else {
        cout << "ERROR: Unable to open file " << dataUserHistory << endl;
    }

    return mapUserHistory;
}

// reading test data
vector<UIPair> getTestData(string dataTest, int userIndex, int itemIndex, char delimiter){
    vector<UIPair> vecTestPairs;
    ifstream ifs(dataTest);
    if(ifs.is_open()){
        while(getline(ifs, line)){
            istringstream iss(line);
            UIPair lp;
            columnNumber = 0;
            while (getline(iss, field, delimiter)){
                if(columnNumber == userIndex){
                    lp.user = stoi(field);
                } else if(columnNumber == itemIndex){
                    lp.item = stoi(field);
                }
                columnNumber++;
            }
            vecTestPairs.push_back(lp);
        }
    } else {
        cout << "ERROR: Unable to open file " << dataTest << endl;
    }

    return vecTestPairs;
}

#endif

