import requests

def make_prediction(text, model_name):
    # URL of the FastAPI endpoint
    url = "http://127.0.0.1:8000/predict/"

    # Data to send in the request
    data = {
        "text": text,
        "model_name": model_name
    }

    # Send POST request to FastAPI server
    try:
        response = requests.post(url, json=data)
        response.raise_for_status()  # Check for HTTP errors
        # Print the response
        print("Response from the API:", response.json())
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
    except Exception as err:
        print(f"An error occurred: {err}")

if __name__ == "__main__":
    while True:
        # Get user input
        text = input("Enter the text to predict sentiment (or type 'exit' to quit): ")

        if text.lower() == 'exit':
            print("Exiting...")
            break

        model_name = input("Enter the model name (e.g., Naive Bayes, Logistic Regression, SVM, Random Forest, XGBoost): ")

        # Make the prediction
        make_prediction(text, model_name)

        # Ask if the user wants to continue
        repeat = input("Do you want to make another prediction? (yes/no): ").strip().lower()
        if repeat != 'yes':
            print("Exiting...")
            break
