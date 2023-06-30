#include "InferenceProperties.h"

Properties::Properties() {
    symmetry = false;
}

void Properties::addToMin(int number) {
    minVector.push_back(number);
}

std::vector<int> Properties::getMinVector() const {
    return minVector;
}

void Properties::setMinVector(const std::vector<int>& newMinVector) {
    minVector = newMinVector;
}

void Properties::addToMax(int number) {
    maxVector.push_back(number);
}

std::vector<int> Properties::getMaxVector() const {
    return maxVector;
}

void Properties::setMaxVector(const std::vector<int>& newMaxVector) {
    maxVector = newMaxVector;
}

void Properties::setSymmetry(bool value) {
    symmetry = value;
}

bool Properties::getSymmetry() const {
    return symmetry;
}
