from models.train import train_model_wrapper, predict_test

if __name__ == "__main__":
    train_model_wrapper()
    submission_df = predict_test()
    print("Done! The submission file has been created.")
