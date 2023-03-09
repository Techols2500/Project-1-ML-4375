// Project_1_Homework 4.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <vector>
#include "Math.h"

#include "Titanic.h"
#include <ctime>
#include "Naive Bayes.h"

using namespace std; 

int main() {

    

    NaiveBayes naive = NaiveBayes();

   
    dataset = vect_Transpose(dataset);
    
    dataset = vect_Transpose(dataset);
    float percentage = 70;
    

  
    naive.fit(training_data);

    
    testing_data = vect_Transpose(testing_data);
    for (int i = 0; i < testing_data.size(); i++)
    {
        auto index = naive.predict(testing_data[i]);
        predicitions.push_back(index);
    }
    testing_data = vect_Transpose(testing_data);
 
    return 0;
}


