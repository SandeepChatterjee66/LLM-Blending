from datasets import load_dataset

try:
    # Load the ConvQuestions dataset
    # The 'pchristm/conv_questions' is the name on Hugging Face Datasets Hub
    dataset = load_dataset("pchristm/conv_questions")

    print("Dataset loaded successfully!")
    print(dataset) # This will show you the splits (train, validation, test)

    # Access a specific split, e.g., the training split
    train_data = dataset['train']
    print("\nFirst example from the training set:")
    print(train_data[0])

    # You can also iterate through the dataset or convert it to a pandas DataFrame
    # df = train_data.to_pandas()
    # print("\nFirst few rows as a Pandas DataFrame:")
    # print(df.head())

except Exception as e:
    print(f"An error occurred while loading the dataset: {e}")
    print("Please ensure you have an active internet connection or try checking the dataset name on Hugging Face Hub.")