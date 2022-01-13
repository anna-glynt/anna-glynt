from feature_names import all_features
import ast


class MapFeatureAnalyzer:
    '''
    This class is called inside the MapAnalyzer (so that the train_log_lines and class_label_list only 
    need to be extracted once instead of searching for each map). Given a map name and index, the class 
    finds the feature importances for the map. The class also gets the range of values that each feature
    takes on in the training targets.

    TO DO:
    Check if extraction target feature values fall in the feature_ranges
    ** Somewhere, we need a way to find the target data in the extraction feature matrix **
    '''
    feature_names = all_features
    importance_threshold = 0.1

    def __init__(self, map_name, map_index, train_log_lines, class_label_list, train_df, target_index, extract_df):

        self.map_name = map_name
        self.map_index = map_index
        self.train_log_lines = train_log_lines
        self.class_label_list = class_label_list

        self.train_df = train_df

        self.target_index = target_index
        self.extract_df = extract_df


    # Call functions and conduct analysis
    def analyze(self):
        print(self.map_name)
        #print(self.train_df.shape)

        self.features, self.importances = self.get_feature_importances(self.train_log_lines)
        self.clean_features = self._strip_map_name_from_features(self.features)
        self.train_feature_ranges = self.get_train_feature_ranges()

        if self.extract_df.size == 0:
            print('Extraction dataframe not found. Map probably did not walk in extraction\n')
        else:
            self.extract_feature_values = self.list_extract_feature_values()
            self.extract_in_train = self.check_extract_features_against_training()
            self.print_results()


    # Get feature importances for each feature used in training
    # This function outputs a list of tuples, with (feature_name, feature_importance)
    # It also breaks the tuples into separate lists of features and importances with the
    # same ordering (i.e. from high to low importance)
    def get_feature_importances(self, train_log_lines):
        for line in train_log_lines:
            if self.map_name.rsplit('-', 3)[0] in line:
                feature_importance_string = line.split('feature importances: ')[1]
                break 
        try:
            feature_importances = ast.literal_eval(feature_importance_string)
            features, importances = zip(*feature_importances)
            return features, importances
        except UnboundLocalError:
            print('Feature importances not found')
        
        
    # For each feature, get a range for the values that it takes on in training targets
    # This function outputs a list of tuples, where the tuples contain the min and max 
    # feature value
    # Note that the tuples in the same order as self.features - that is, ordered from high
    # to low feature importance
    def get_train_feature_ranges(self):
        train_feature_ranges = []
        train_df = self.train_df.iloc[self.class_label_list]
        for feature in self.features:
            value_list = train_df[feature].tolist()
            # Note: this function uses the range. It  would be easy to use an average instead
            range = (min(value_list), max(value_list))
            train_feature_ranges.append(range)
        return train_feature_ranges


    # List feature values in extraction
    def list_extract_feature_values(self):
        extract_target_df = self.extract_df.iloc[self.target_index]
        return [extract_target_df[feature] for feature in self.features]


    # Check whether extraction values fall within range of training features
    def check_extract_features_against_training(self):
        extract_in_train = []
        for ii in range(len(self.features)):
            test1 = self.extract_feature_values[ii] >= self.train_feature_ranges[ii][0]
            test2 = self.extract_feature_values[ii] <= self.train_feature_ranges[ii][1]
            extract_in_train.append(bool(test1 and test2))
        return extract_in_train


    # Print results
    def print_results(self):
        print(f'Training/extraction feature value mismatches for features with importance > {self.importance_threshold}:')
        for ii in range(len(self.features)):
            if (not self.extract_in_train[ii] and self.importances[ii] > self.importance_threshold):
                print(f'Feature: {self.clean_features[ii]}. Importance: {self.importances[ii]}. Training range: {self.train_feature_ranges[ii]}. Extract feature value: {self.extract_feature_values[ii]}')
        print('\n')


    # Get just the standard feature name (without the map name) from a list of features
    def _strip_map_name_from_features(self, feature_list):
        new_features = []
        for feature in feature_list:
            for string in self.feature_names:
                if feature.endswith(string):
                    new_features.append(string)
                    break
        return new_features
