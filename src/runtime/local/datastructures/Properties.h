#ifndef PROPERTIES_H
#define PROPERTIES_H

#include <vector>
#include <iostream>

struct Properties {
    std::vector<double> minMax;
    std::vector<int> vector2;
    bool symmetry;
    bool unique;
    int value;

    /*
        Histograms definition - equi-width

        @histograms - the values of each interval
        @selectivity - after performing a filter op get the percentage of data.
    */
    std::vector<int> histograms;
    int numberOfBuckets;
    double selectivity;

    /*
        Empty constructor
    */
    Properties(): 
        symmetry(false),
        numberOfBuckets(0), 
        selectivity(0.0)
        {}

    /*
        Histogram constructor
    */
    Properties(std::vector<int> newHistograms, int newNumberOfBuckets, double newSelectivity) :
        histograms(newHistograms),
        numberOfBuckets(newNumberOfBuckets),
        selectivity(newSelectivity)
        {}

    /*
        Check if any of the properties changed.
        If yes return a new properties structure.
    */
    Properties createIfChanged(std::vector<int> newHistograms, int newNumberOfBuckets, double newSelectivity) const {

        if (
            std::equal(newHistograms.begin(), newHistograms.end(), histograms.begin()) &&
            newSelectivity == selectivity &&
            numberOfBuckets == newNumberOfBuckets
        ) 
            return *this;
        
        return Properties(newHistograms, newNumberOfBuckets ,newSelectivity);

    }

};

#endif //PROPERTIES_H