'''from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from model import Encoder, DecoderHELLOW

app = Flask(__name__)
CORS(app)  # This will allow cross-origin requests.

# Define model parameters.
lstm_cells = 512
embedding_dim = 256
vocab_size = 5000  # Adjust this according to your actual vocab size.
MODEL_PATH = "my_model_weights.h5"

# Check if the model file exists.
if not os.path.exists(MODEL_PATH):
    raise ValueError(f"Model file '{MODEL_PATH}' not found.")

# Initialize the Encoder and Decoder.
encoder = Encoder(lstm_cells, embedding_dim, vocab_size)
decoder = Decoder(lstm_cells, embedding_dim, vocab_size)

# Initialize the ChatBot.
chatbot = RealTimeChatBot(encoder, decoder)
chatbot.load_weights(MODEL_PATH)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_message = data.get("message", "")
    
    # Use your RealTimeChatBot instance for getting the response.
    chatbot_response = chatbot.call(user_message)

    return jsonify({"response": chatbot_response})

if __name__ == '__main__':
    app.run(debug=True)'''

from flask import Flask, request, jsonify
import tensorflow as tf

app = Flask(__name__)

# Load the model
model_path = "D:/3 rd Year/Y4S1/Reserch/Chatbot/ckpt1"
model = tf.keras.models.load_model(model_path)

# For the sake of this example, I'm defining a mock tokenizer.
# Replace this with your actual tokenizer.
class MockTokenizer:
    index_word = {1: 'hello', 2: 'world', 3: 'EOS'}

# Use your tokenizer here.
tokenizer = MockTokenizer()

def post_process(prediction, tokenizer):
    # Convert predicted token IDs to words.
    token_ids = tf.argmax(prediction, axis=-1).numpy()[0]
    words = [tokenizer.index_word.get(token_id, '?') for token_id in token_ids if token_id != 0]  # Exclude padding tokens.

    # Stop at EOS token if present.
    if 'EOS' in words:
        words = words[:words.index('EOS')]

    # Join words to form the final sentence.
    return ' '.join(words)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    encoder_input = data['encoder_input']
    decoder_input = data['decoder_input']

    prediction = model.predict([encoder_input, decoder_input])
    response = post_process(prediction, tokenizer)

    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)




