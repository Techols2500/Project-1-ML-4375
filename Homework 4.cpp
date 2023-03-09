// Homework 4.cpp : This file contains the 'main' function. Program execution begins and ends there.
//
#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <math.h>
#include <numeric>
#include <sstream>
#include <chrono>
#include <map>
#include <list>

using namespace std;
using namespace std::chrono;

vector<vector<string>> getContents(string fileName) {
    //read file content to vector
    vector<vector<string>> content;
    vector<string> row;
    string line;
    string word;

    fstream file(fileName, ios::in);

    if (file.is_open()) {
        while (getline(file, line)) {
            row.clear();
            stringstream str(line);
            while (getline(str, word, ',')) {
                row.push_back(word);
            }
            content.push_back(row);
        }
        file.close();
    }
    else {
        cout << "Could not open the file : " << fileName << endl;
    }

    return content;
}

void printFileContents(vector<vector<string>> contents, int lines) {
    cout << "Rows : " << contents.size() << endl;
    cout << "Cols : " << contents[0].size() << endl;
    if (lines == -1) {
        lines = contents.size();
    }
    for (int i = 0; i < lines; i++) {
        for (int j = 0; j < contents[i].size(); j++) {
            cout << contents[i][j] << "\t" << flush; //FIXME: flush doesn't seem to be working in run mode but ok in debug mode.
        }
        std::cout << std::endl;
    }
}

vector<vector<string>> data_cleanup(vector<vector<string>> data, int columnToRemove) {
    for (auto& r : data) {
        r.erase(r.begin() + columnToRemove);
    }
    return data;
}

vector<vector<double>> convert_data_to_double(vector<vector<string>> data) {
    int rows = data.size(); //number of rows
    int cols = data[0].size(); // number of columns
    vector<vector<double>> ddata;
    for (int i = 0; i < rows; i++) {
        vector<double> row;
        for (int j = 0; j < cols; j++) {
            string val = data[i].at(j);
            row.push_back(stod(val));
        }
        ddata.push_back(row);
    }
    return ddata;
}

tuple<vector<vector<double>>, vector<double>, vector<vector<double>>, vector<double>>
doTrainTestSplit(vector<vector<double>> dataset, double train_size) {

    int rows = dataset.size();
    int cols = dataset[0].size();

    int train_rows = round(train_size * rows);
    int test_rows = rows - train_rows;

    vector<vector<double>> X_train;
    vector<double> y_train;
    vector<vector<double>> X_test;
    vector<double> y_test;

    //TRAIN SET
    for (int i = 0; i < train_rows; i++) {
        vector<double> x_train_row;
        for (int j = 0; j < cols; j++) {
            double val = dataset[i].at(j);
            if (j != 1)
                x_train_row.push_back(val);
            else
                y_train.push_back(val);
        }
        X_train.push_back(x_train_row);
    }

    //TEST SET
    for (int i = train_rows; i < rows; i++) {
        vector<double> x_test_row;
        for (int j = 0; j < cols; j++) {
            double val = dataset[i].at(j);
            if (j != 1)
                x_test_row.push_back(val);
            else
                y_test.push_back(val);
        }
        X_test.push_back(x_test_row);
    }

    return std::make_tuple(X_train, y_train, X_test, y_test);
}

double gaussian(double x, double mu, double sigma) {
    double expr1 = 1.0 / sqrt(2 * M_PI);
    double expr2 = pow((x - mu) / sigma, 2) * -0.5;
    return (expr1 / sigma) * exp(expr2);
}

double possible_labels[2] = { 0,1 }; // possible labels are SURVIVED (1) or NOT SURVIVED (0)
std::map<double, double> find_label_count(const vector<double>& labels) {

    std::map<double, double> priors;
    for (const auto& label : possible_labels) {
        priors.insert(std::make_pair(label, 0.0));
    }

    //start counting from here.
    for (const auto& label : labels) {
        auto iter = priors.find(label);
        if (iter != priors.end()) {
            iter->second++;
        }
    }
    return priors;
}

struct stats {
    double mu;
    double sigma;
};

std::map <double, vector<stats>> m_stats_data;
void calc_stats(const vector<vector<double>>& data, const vector<double>& labels) {

    std::map<double, vector<vector<double>>> data_table;
    std::map<double, vector<vector <double>>> cond_probs;
    vector<stats> musigma;

    if (labels.size() != data.size()) {
        throw "FATAL: training data and labels are of different lengths";
    }

    for (int i = 0; i < data[0].size(); i++) {
        musigma.push_back(stats{ 0.0,0.0 });
    }

    for (const auto& label : possible_labels) {
        cond_probs.insert(std::make_pair(label, vector<vector<double>>()));
        data_table.insert(std::make_pair(label, vector<vector<double>>()));
        m_stats_data.insert(std::make_pair(label, vector<stats>{musigma}));
    }


    //find mean and stddev of data w.r.t. labels.
    for (int indx = 0; indx < labels.size(); indx++) {
        data_table[labels[indx]].push_back(data[indx]);
        // perform 2.
        for (int state = 0; state < data[indx].size(); state++)
        {
            m_stats_data[labels[indx]][state].mu += data[indx][state];
        }
    }

    // complete 2.
    auto labelCount = find_label_count(labels);
    for (const auto& label : possible_labels)
    {
        for (int state = 0; state < data[0].size(); state++)
        {
            m_stats_data[label][state].mu /= labelCount[label];
        }
    }
    // 3.a. compute stddev for each label and each state
    for (const auto& label : possible_labels)
    {
        //compute expression = sum{pow((x_i - xBar),2)}
        for (const auto& datum : data_table[label])
        {
            for (int state = 0; state < datum.size(); state++)
            {
                m_stats_data[label][state].sigma += pow((datum[state] - m_stats_data[label][state].mu), 2);
            }
        }
    }

    // 3.b. compute stddev = sqrt(expression/N)
    for (const auto& label : possible_labels) {
        for (int state = 0; state < data[0].size(); state++) {
            m_stats_data[label][state].sigma /= labelCount[label];
            m_stats_data[label][state].sigma = sqrt(m_stats_data[label][state].sigma);
        }
    }

}

std::map<double, double> m_prior_probs;
void train(const vector<vector<double>>& data, const vector<double>& labels) {

    m_prior_probs = find_label_count(labels);
    for (const auto& label : possible_labels) {
        m_prior_probs[label] /= labels.size();
    }
    //compute statistics for each label
    calc_stats(data, labels);
}

double predict(const vector<double>& sample) {
    std::map<double, double> naive_bayes_classifier;
    for (const auto& label : possible_labels)
    {
        vector<double> probs;
        double product = m_prior_probs[label];
        for (int state = 0; state < sample.size(); state++) {
            double prob = gaussian(sample[state], m_stats_data[label][state].mu, m_stats_data[label][state].sigma);
            product *= prob;
        }
        naive_bayes_classifier[label] = product;
    }
    //find argmax and return
    return std::max_element(naive_bayes_classifier.begin(), naive_bayes_classifier.end(),
        [](const std::pair<double, double>& a, const std::pair<double, double>& b) {return (a.second < b.second); })->first;
}

int main() {

    //1. get titatic file contents
    vector<vector<string>> contents_original = getContents("titanic_project.csv");
    if (contents_original.size() == 0) {
        cout << "No file contents " << endl;
        return EXIT_FAILURE;
    }

    //2. data clean up
    int rows = contents_original.size(); //number of rows
    int cols = contents_original[0].size(); // number of columns
    vector<vector<string>> dataset = data_cleanup(contents_original, 0);// first columns is not really required.
    printFileContents(dataset, 10);

    //3. remove first row containing column names.
    dataset.erase(dataset.begin() + 0);

    //4. covert all data into double.
    vector<vector<double>> double_dataset;
    double_dataset = convert_data_to_double(dataset);

    //5. split into train and test dataset.
    vector<vector<double>> x_train;
    vector<double> y_train;
    vector<vector<double>> x_test;
    vector<double> y_test;

    tuple<vector<vector<double>>, vector<double>, vector<vector<double>>, vector<double>> split_data =
        doTrainTestSplit(double_dataset, 0.86); // 900 Train and Remaining Test Records will be created.
    tie(x_train, y_train, x_test, y_test) = split_data;

    //6 Train Gaussian
    auto start = high_resolution_clock::now(); // to caculate the time taken
    train(x_train, y_train);
    auto stop = high_resolution_clock::now();
    duration<double> elapsed_sec = stop - start;
    cout << "Time taken for training :" << elapsed_sec.count() << endl;

    //7 Predict
    /*
    cout << "X_train number of elements " << x_test.size() << endl;
    cout << "X_train element size " << x_test[0].size() << endl;
    cout << "Y_test number of elements " << y_test.size() << endl;
    */

    int score = 0;
    for (int i = 0; i < x_test.size(); ++i) {
        vector<double> coords = x_test[i];
        double predicted = predict(coords);
        if (predicted == (y_test[i])) {
            score += 1;
        }
    }

    float fraction_correct = float(score) / y_test.size();
    cout << "accuracy = " << (100 * fraction_correct) << endl;

    return 0;
}

