#ifndef INFERENCEPROPERTIES_H
#define INFERENCEPROPERTIES_H

#include <vector>

class Properties {
private:
    std::vector<int> minVector;
    std::vector<int> maxVector;
    bool symmetry;
    
public:
    Properties();
    void addToMin(int number);
    std::vector<int> getMinVector() const;
    void setMinVector(const std::vector<int>& newMinVector);
    void addToMax(int number);
    std::vector<int> getMaxVector() const;
    void setMaxVector(const std::vector<int>& newMaxVector);
    void setSymmetry(bool value);
    bool getSymmetry() const;
};

#endif  // INFERENCEPROPERTIES_H
