from flask import Flask, request, jsonify, render_template
app = Flask(__name__)



# Import and load your machine learning model here
# Example: 
# from your_module import your_model
# your_model.load_weights('model_weights.h5')

your_model = tf.keras.saving.load_model(r'C:\Users\farra\OneDrive\Desktop\Machine_Learning\InceptionV3_Vegetabletoptune_1epochs')
#sam.load_weights('/Users/marmik/Downloads/sam_vit_b_01ec64.pth')



@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
        try:
    

    # Get the input data as JSON from the POST request
            data = request.get_json()

        # Validate and preprocess the input data if needed
            input_data = preprocess_data(data)

        # Call your machine learning model for inference
            prediction = your_model.predict(input_data)

        # Format the prediction result as needed
            result = format_prediction(prediction)
            return jsonify({'result': result})

        except Exception as e:
            return jsonify({'error': str(e)})
            return {'result': 'Prediction result'}
        



# Front-end API endpoint
@app.route('/frontend_api', methods=['POST'])
def frontend_api():
    try:
        data = request.get_json()
        # Process data, interact with your model, etc.
        # You can add your specific logic here based on the front-end requirements
        result = {'message': 'Front-end API call successful'}
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)})
                       

if __name__ == '__main__':
    app.run(debug=True)





