#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <cmath>
#include <limits>
#include <algorithm>
#include <iomanip>
#include <chrono>  // used for measuring runtime

using namespace std;
using namespace std::chrono;

// function to perform leave-one-out cross validation on the data with a given feature set
double leave_one_out_cross_validation(const vector<vector<double>>& data, const vector<int>& current_set) {
    int number_correctly_classified = 0;
    int num_samples = data.size();

    // iterate through each sample in the dataset
    for (int i = 0; i < num_samples; i++) {
        vector<double> object_to_classify(data[i].size(), 0.0);
        
        // select the features specified in current_set for the object to classify
        for (int feature : current_set) {
            object_to_classify[feature] = data[i][feature];
        }
        double label_object_to_classify = data[i][0];  // class label of the current sample

        double nearest_neighbor_distance = numeric_limits<double>::infinity();
        int nearest_neighbor_location = -1;

        // find the nearest neighbor
        for (int k = 0; k < num_samples; k++) {
            if (k != i) {  // skip the current sample
                vector<double> neighbor(data[k].size(), 0.0);
                
                // select features for the neighbor
                for (int feature : current_set) {
                    neighbor[feature] = data[k][feature];
                }

                // calculate Euclidean distance between the object and the neighbor
                double distance = 0.0;
                for (size_t j = 0; j < current_set.size(); j++) {
                    distance += pow(object_to_classify[current_set[j]] - neighbor[current_set[j]], 2);
                }
                distance = sqrt(distance);  

                // update the nearest neighbor if the current one is closer
                if (distance < nearest_neighbor_distance) {
                    nearest_neighbor_distance = distance;
                    nearest_neighbor_location = k;
                }
            }
        }

        // check if the nearest neighbor's label matches the object's label
        if (nearest_neighbor_location != -1) {
            double nearest_neighbor_label = data[nearest_neighbor_location][0];
            if (label_object_to_classify == nearest_neighbor_label) {
                number_correctly_classified++;
            }
        }
    }
    
    return static_cast<double>(number_correctly_classified) / num_samples;
}

// function to perform forward feature selection
void feature_search_demo(const vector<vector<double>>& data) {
    vector<int> current_set_of_features;
    int num_features = data[0].size() - 1;  // exclude the class label

    cout << "Beginning search." << endl;
    double best_overall_accuracy = 0.0;
    vector<int> best_feature_set;

    // for each feature level
    for (int i = 0; i < num_features; i++) {
        int feature_to_add_at_this_level = -1;
        double best_so_far_accuracy = 0.0;

        // try adding each unselected feature
        for (int k = 1; k <= num_features; k++) {
            if (find(current_set_of_features.begin(), current_set_of_features.end(), k) == current_set_of_features.end()) {
                vector<int> new_set = current_set_of_features;
                new_set.push_back(k);  // add the feature to the set
                double accuracy = leave_one_out_cross_validation(data, new_set);
                
                // display the current feature set and its accuracy
                cout << "   Using feature(s) {";
                for (size_t j = 0; j < new_set.size(); j++) {
                    cout << new_set[j] << (j < new_set.size() - 1 ? ", " : "");
                }
                cout << "} accuracy is " << fixed << setprecision(1) << accuracy * 100 << "%" << endl;

                // update the best feature to add at this level
                if (accuracy > best_so_far_accuracy) {
                    best_so_far_accuracy = accuracy;
                    feature_to_add_at_this_level = k;
                }
            }
        }

        // add the best feature to the set
        if (feature_to_add_at_this_level != -1) {
            cout << "Feature set {" << feature_to_add_at_this_level << "} was best, accuracy is " << fixed << setprecision(1) << best_so_far_accuracy * 100 << "%" << endl;
            current_set_of_features.push_back(feature_to_add_at_this_level);
            if (best_so_far_accuracy > best_overall_accuracy) {
                best_overall_accuracy = best_so_far_accuracy;
                best_feature_set = current_set_of_features;
            }
        }
    }

    cout << "Finished search. The best feature subset is {";
    for (size_t j = 0; j < best_feature_set.size(); j++) {
        cout << best_feature_set[j] << (j < best_feature_set.size() - 1 ? ", " : "");
    }
    cout << "}, which has an accuracy of " << fixed << setprecision(1) << best_overall_accuracy * 100 << "%" << endl;
}

// function to perform backward elimination feature selection
void backward_elimination_demo(const vector<vector<double>>& data) {
    vector<int> current_set_of_features;
    int num_features = data[0].size() - 1;  // exclude the class label

    // start with all features
    for (int i = 1; i <= num_features; i++) {
        current_set_of_features.push_back(i);
    }

    cout << "Beginning backward elimination." << endl;

    // evaluate accuracy with all features
    double best_overall_accuracy = leave_one_out_cross_validation(data, current_set_of_features);
    vector<int> best_feature_set = current_set_of_features;

    // perform backward elimination
    while (current_set_of_features.size() > 0) {
        double best_so_far_accuracy = 0.0;
        int feature_to_remove = -1;
        vector<int> best_temp_set = current_set_of_features;

        // try removing each feature and check the accuracy
        for (size_t i = 0; i < current_set_of_features.size(); i++) {
            vector<int> new_set = current_set_of_features;
            new_set.erase(new_set.begin() + i);  // Remove the feature at position i

            double accuracy = leave_one_out_cross_validation(data, new_set);
            
            // display the current feature set and its accuracy
            cout << "Using feature(s) {";
            for (size_t j = 0; j < new_set.size(); j++) {
                cout << new_set[j] << (j < new_set.size() - 1 ? ", " : "");
            }
            cout << "} accuracy is " << fixed << setprecision(1) << accuracy * 100 << "%" << endl;

            // update the best feature to remove
            if (accuracy > best_so_far_accuracy || feature_to_remove == -1) {
                best_so_far_accuracy = accuracy;
                feature_to_remove = i;
                best_temp_set = new_set;
            }
        }

        cout << "Removing feature " << current_set_of_features[feature_to_remove] << " for best accuracy of " << fixed << setprecision(1) << best_so_far_accuracy * 100 << "%" << endl;
        current_set_of_features = best_temp_set;
        
        // Update the best feature set if accuracy improved
        if (best_so_far_accuracy > best_overall_accuracy) {
            best_overall_accuracy = best_so_far_accuracy;
            best_feature_set = current_set_of_features;
        }
    }

    // output the best feature subset and its accuracy
    cout << "Finished search. The best feature subset is {";
    for (size_t j = 0; j < best_feature_set.size(); j++) {
        cout << best_feature_set[j] << (j < best_feature_set.size() - 1 ? ", " : "");
    }
    cout << "}, which has an accuracy of " << fixed << setprecision(1) << best_overall_accuracy * 100 << "%" << endl;
}

// function to read data from a file and store it in a 2D vector
vector<vector<double>> read_data_from_file(const string& filename) {
    vector<vector<double>> data;
    ifstream file(filename);
    string line;

    if (!file) {
        cerr << "Error opening file: " << filename << endl;
        exit(1);
    }

    // read each line of the file and parse it into a vector of doubles
    while (getline(file, line)) {
        stringstream ss(line);
        vector<double> row;
        double value;
        while (ss >> value) {
            row.push_back(value);
        }
        if (!row.empty()) {
            data.push_back(row);
        }
    }

    file.close();
    return data;
}

int main() {
    string filename;
    string algorithmNum;
    cout << "Welcome to Tayvin Nguyen's Feature Selection Algorithm." << endl;
    cout << "Type in the name of the file to test: ";
    cin >> filename;
    cout << endl;

    // read the dataset from the specified file
    vector<vector<double>> data = read_data_from_file(filename);

    cout << "Type in the number of the algorithm you want to run." << endl;
    cout << "   1) Forward Selection" << endl;
    cout << "   2) Backward Elimination" << endl;

    cin >> algorithmNum;

    // get number of features 
    int num_features = data[0].size() - 1;
    int num_instances = data.size();

    // start measuring time
    auto start = high_resolution_clock::now();

    // evaluate accuracy with all features
    vector<int> all_features;
    for (int i = 1; i <= num_features; i++) {
        all_features.push_back(i);
    }
    double initial_accuracy = leave_one_out_cross_validation(data, all_features);
    
    cout << "This dataset has " << num_features << " features (not including the class attribute), with "
         << num_instances << " instances." << endl;
    cout << "Running nearest neighbor with all " << num_features << " features, using \"leaving-one-out\" "
         << "evaluation, I get an accuracy of " 
         << fixed << setprecision(1) << initial_accuracy * 100 << "%" << endl;

    // run the selected feature selection algorithm
    if (algorithmNum == "1") {
        feature_search_demo(data); 
    } else if (algorithmNum == "2") {
        backward_elimination_demo(data);
    } else {
        cout << "Not a valid choice." << endl;
    }
    
    // end timing
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<seconds>(end - start); 

    cout << "Runtime: " << duration.count() << " seconds" << endl;

    return 0;
}
