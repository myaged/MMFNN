#ifndef TUPLE_H
#define TUPLE_H

/*
    Tuple class

    Part of MMFNN guiding code. Provided as is.
    Tested with C++11 (g++ ver. 4.9.4)
 */

class Tuple {

    private:
        unsigned int userId;
        unsigned int itemId;
        unsigned int relevance;

    public:
        Tuple(unsigned int userId, unsigned int itemId, unsigned int relevance){
            this->userId = userId;
            this->itemId = itemId;
            this->relevance = relevance;
        }

        unsigned int getUserId(){
            return this->userId;
        }

        unsigned int getItemId(){
            return this->itemId;
        }

        unsigned int getRelevance(){
            return this->relevance;
        }
};

#endif
