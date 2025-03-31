import gradio as gr
import pickle

# Load the model and vectorizer
with open("model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("vectorizer.pkl", "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Define the prediction function
def predict_sentiment(text):
    text_vectorized = vectorizer.transform([text])
    prediction = model.predict(text_vectorized)
    sentiment = "Positive" if prediction == 1 else "Negative"
    return sentiment


custom_css = """
body, .gradio-container, p, h1, h2, h3, h4, h5, h6, label {
    color: black !important;
}
body { background-color: #FFF9C4; } 
h1, h2 { color: #333333; font-weight: bold; } /* Dark title */
.gradio-container { background: #FFF9C4; border-radius: 10px; padding: 20px; } /* Pastel pink container */
input, textarea { background: #AFCBFF; border: none; border-radius: 5px; } /* Light pastel blue */
button { background: #B5EAD7; color: black; border: none; border-radius: 8px; padding: 10px; font-size: 16px; } /* Pastel green */
"""


with gr.Blocks(css=custom_css) as app:
    gr.Markdown("<h1 style='text-align:center; color:black;'>Sentiment Analysis 💬</h1>")
    gr.Markdown("<p style='text-align:center; color:black;'>Enter a sentence to check if it's positive or negative.</p>")


    with gr.Row():
        input_text = gr.Textbox(label="Your Text", placeholder="Type something here...", lines=2)
        predict_button = gr.Button("Analyze Sentiment 🚀")

    output_text = gr.Textbox(label="Prediction", interactive=False)

    predict_button.click(fn=predict_sentiment, inputs=input_text, outputs=output_text)

app.launch()