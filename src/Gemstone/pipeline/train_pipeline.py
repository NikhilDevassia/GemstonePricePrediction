
def train_pipeline(ingest_data, clean_data, model_train, evaluate_model):
    """
    Train a pipeline consisting of data ingestion, data cleaning,
    model training, and model evaluation.

    Args:
        ingest_data (Class): The class responsible for data ingestion.
        clean_data (Class): The class responsible for data cleaning.
        model_train (Class): The class responsible for model training.
        evaluate_model (Class): The class responsible for model evaluation.

    Returns:
        tuple: A tuple containing the mean absolute error (mae),
               mean squared error (mse), root mean squared error (rmse),
               and R-squared score (r2_score).
    """
    # Data ingestion
    data = ingest_data.initiate_ingest_data()

    # Data cleaning and transformation
    train_arr, test_arr, _ = clean_data.clean_data_and_transform(data=data)

    # Model training
    X_train, X_test, y_train, y_test, best_model = model_train.initiate_model_training(train_arr, test_arr)

    # Model evaluation
    mae, mse, rmse, r2_score = evaluate_model.evaluate_single_model(X_train, X_test, y_train, y_test, best_model)

    return mae, mse, rmse, r2_score
