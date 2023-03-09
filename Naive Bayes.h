#ifndef  NAIVEBAYES_H
#define NAIVEBAYES_H 

typedef struct class_summary
{
    
    float class_prob;

} class_summary;

class NaiveBayes
{
private:
   

public:
    void fit();
    int  predict(const );
};

class_summary calculate_Class_Summary(std::vector<std::vector<float>> dataset, float class_label);
float prob_By_Summary(const std::vector<float>& test_data, const class_summary& summary);
#endif

