def k_folding_data(data,label,k):
    '''
    # Splits data into k folds for cross-validation.
    
    # Parameters:
    # data (list or array-like): The dataset to be split.
    # label (list or array-like): Corresponding labels for the dataset.
    # k (int): Number of folds.
    
    # Returns:
    # list of tuples: Each tuple contains training and validation data and labels for a fold.
    # '''
    # fold_size = len(data) // k
    # folds = []
    
    # for i in range(k):
    #     val_data = data[i*fold_size:(i+1)*fold_size]
    #     val_label = label[i*fold_size:(i+1)*fold_size]
        
    #     train_data = data[:i*fold_size] + data[(i+1)*fold_size:]
    #     train_label = label[:i*fold_size] + label[(i+1)*fold_size:]
        
    #     folds.append((train_data, train_label, val_data, val_label))
    
    return folds