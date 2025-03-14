import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
from tensorflow.keras.models import load_model
import pickle
import os
from googletrans import Translator

# Initialize translator
translator = Translator()

# Define language options with names in their native script
LANGUAGES = {
    # Indian Languages
    'English': 'en',
    'हिंदी (Hindi)': 'hi',
    'বাংলা (Bengali)': 'bn',
    'తెలుగు (Telugu)': 'te',
    'मराठी (Marathi)': 'mr',
    'தமிழ் (Tamil)': 'ta',
    'ગુજરાતી (Gujarati)': 'gu',
    'ಕನ್ನಡ (Kannada)': 'kn',
    'മലയാളം (Malayalam)': 'ml',
    'ਪੰਜਾਬੀ (Punjabi)': 'pa',
    'ଓଡ଼ିଆ (Odia)': 'or',
    'অসমীয়া (Assamese)': 'as',
    
    # International Languages
    '中文 (Chinese)': 'zh-cn',
    'Español (Spanish)': 'es',
    'العربية (Arabic)': 'ar',
    'Français (French)': 'fr',
    'Русский (Russian)': 'ru',
    'Português (Portuguese)': 'pt',
    'Bahasa Indonesia': 'id',
    '日本語 (Japanese)': 'ja',
    'Deutsch (German)': 'de',
    '한국어 (Korean)': 'ko'
}

# Set page config
st.set_page_config(page_title="Agriculture Assistant", layout="wide")

# Define disease classes
DISEASE_CLASSES = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

def translate_text(text, target_lang='en'):
    """Translate text to target language"""
    try:
        if target_lang == 'en':
            return text
        translation = translator.translate(text, dest=target_lang)
        return translation.text
    except Exception as e:
        st.warning(f"Translation error: {str(e)}")
        return text

def load_models():
    """Load ML models and class labels"""
    try:
        # Load disease detection model
        disease_model = tf.keras.models.load_model("disease_model.h5")
        
        # Load class indices from training
        train_dir = '/Users/vinuthnarajeswari/Desktop/archive/New Plant Diseases Dataset(Augmented)/train'
        temp_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255)
        temp_generator = temp_datagen.flow_from_directory(
            train_dir,
            target_size=(128, 128),
            batch_size=1,
            class_mode='categorical'
        )
        class_indices = temp_generator.class_indices
        disease_labels = {v: k for k, v in class_indices.items()}
        
        return disease_model, disease_labels
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None

def predict_disease(image_file, model, disease_labels):
    """Predict plant disease from image"""
    try:
        img = Image.open(image_file)
        img = img.resize((128, 128))
        img_array = np.array(img)
        
        if len(img_array.shape) == 2:
            img_array = np.stack((img_array,)*3, axis=-1)
        elif img_array.shape[2] == 4:
            img_array = img_array[:,:,:3]
        
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class])
        
        disease_name = disease_labels[predicted_class]
        
        return disease_name, confidence
    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")
        return None, None

def load_crop_model():
    """Load crop recommendation model and preprocessing tools"""
    try:
        with open('crop_recommendation_model.pkl', 'rb') as f:
            crop_model = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
        
        return crop_model, scaler, label_encoder
    except Exception as e:
        st.error(f"Error loading crop recommendation tools: {str(e)}")
        return None, None, None

def predict_crop(soil_features, model, scaler, label_encoder):
    """Predict suitable crop from soil parameters"""
    try:
        features_scaled = scaler.transform([soil_features])
        prediction = model.predict(features_scaled)
        crop_name = label_encoder.inverse_transform(prediction)
        return crop_name[0]
    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")
        return None

def main():
    # Language selector in sidebar with search functionality
    with st.sidebar:
        selected_language = st.selectbox(
            "Select Language / भाषा चुनें / ভাষা নির্বাচন করুন / భాష ఎంచుకోండి",
            list(LANGUAGES.keys()),
            format_func=lambda x: x  # Display native script
        )
        lang_code = LANGUAGES[selected_language]
    
    # Translate main titles
    st.title("🌱 " + translate_text("Agriculture Assistant", lang_code))
    st.subheader(translate_text("Your Smart Farming Companion", lang_code))
    
    # Load models
    disease_model, disease_labels = load_models()
    crop_model, scaler, label_encoder = load_crop_model()
    
    if not disease_model or not disease_labels:
        st.error(translate_text("Failed to load model or class labels.", lang_code))
        return
    
    with st.sidebar:
        st.title(translate_text("Services", lang_code))
        service = st.radio("", [
            translate_text("Disease Detection", lang_code),
            translate_text("Crop Recommendation", lang_code)
        ])
    
    if service == translate_text("Disease Detection", lang_code):
        st.header(translate_text("Disease Detection", lang_code))
        
        # Add help text in selected language
        st.markdown(translate_text(
            "Upload a clear image of the plant leaf to detect diseases. " +
            "Make sure the leaf is well-lit and centered in the image.", 
            lang_code
        ))
        
        uploaded_file = st.file_uploader(
            translate_text("Choose an image...", lang_code),
            type=['jpg', 'png', 'jpeg']
        )
        
        if uploaded_file:
            col1, col2 = st.columns(2)
            with col1:
                st.image(uploaded_file, caption=translate_text("Uploaded Image", lang_code))
                
                if st.button(translate_text("Detect Disease", lang_code)):
                    with st.spinner(translate_text("Analyzing...🔄", lang_code)):
                        disease_name, confidence = predict_disease(uploaded_file, disease_model, disease_labels)
                        
                        if disease_name:
                            with col2:
                                st.success(translate_text("Analysis Complete!✅", lang_code))
                                st.write(f"**{translate_text('Predicted Disease:', lang_code)}** " + 
                                       f"{translate_text(disease_name, lang_code)}")
                                st.write(f"**{translate_text('Confidence:', lang_code)}** {confidence*100:.2f}%")

                                st.subheader(translate_text("💊 Treatment Recommendations:", lang_code))
                                if "healthy" in disease_name.lower():
                                    healthy_message = translate_text(
                                        "The plant appears to be healthy! Continue with regular care:",
                                        lang_code
                                    )
                                    st.write(healthy_message)
                                    
                                    care_tips = [
                                        "Regular watering 💧",
                                        "Proper fertilization 🌱",
                                        "Regular monitoring 👀",
                                        "Maintain good air circulation 🌬️",
                                        "Watch for early signs of stress 🔍"
                                    ]
                                    
                                    for tip in care_tips:
                                        st.write("• " + translate_text(tip, lang_code))
                                else:
                                    st.write(translate_text("Recommended actions:", lang_code))
                                    
                                    treatment_steps = [
                                        "⚠️ Isolate affected plants",
                                        "✂️ Remove affected leaves",
                                        "🧪 Apply appropriate fungicide/pesticide",
                                        "💨 Improve air circulation",
                                        "🌡️ Monitor temperature and humidity",
                                        "💧 Adjust watering practices"
                                    ]
                                    
                                    for step in treatment_steps:
                                        st.write("• " + translate_text(step, lang_code))
                        else:
                            st.error(translate_text(
                                "Error: Unable to determine disease class. Please try again with a clearer image.",
                                lang_code
                            ))
    
    else:  # Crop Recommendation
        st.header(translate_text("Crop Recommendation 🌾", lang_code))
        st.write(translate_text("Enter soil parameters to get crop recommendations 📊", lang_code))
        
        col1, col2 = st.columns(2)
        
        with col1:
            nitrogen = st.number_input("Nitrogen (N)", 0, 150, 143)
            phosphorus = st.number_input("Phosphorus (P)", 0, 140, 69)
            potassium = st.number_input("Potassium (K)", 0, 250, 217)
            electrical_conductivity = st.number_input("Electrical Conductivity (EC)", 0.0, 100.0, 0.58)
            ph = st.number_input("pH", 0.0, 14.0, 5.9)
            
        with col2:
            sulfur = st.number_input("Sulfur (S)", 0.0, 300.0, 0.23)
            copper = st.number_input("Copper (Cu)", 0.0, 300.0, 10.2)
            iron = st.number_input("Iron (Fe)", 0.0, 300.0, 116.35)
            manganese = st.number_input("Manganese (Mn)", 0.0, 300.0, 59.96)
            zinc = st.number_input("Zinc (Zn)", 0.0, 300.0, 54.85)
            boron = st.number_input("Boron (B)", 0.0, 300.0, 21.29)
        
        if st.button("Get Recommendation 🎯"):
            soil_features = [
                nitrogen, phosphorus, potassium, electrical_conductivity,
                ph, sulfur, copper, iron, manganese, zinc, boron
            ]
            
            with st.spinner("Analyzing soil parameters..."):
                crop = predict_crop(soil_features, crop_model, scaler, label_encoder)
                if crop:
                    st.markdown(f"""
                        <div class="success-message" style="text-align: center; margin-top: 20px;">
                            <h3>Recommended Crop: {crop} 🌱</h3>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Add crop specific tips
                    st.subheader(translate_text("Cultivation Tips:", lang_code))
                    tips = [
                        "Best planting season",
                        "Water requirements",
                        "Fertilizer recommendations",
                        "Common pests and diseases",
                        "Harvesting time"
                    ]
                    for tip in tips:
                        st.write("• " + translate_text(tip, lang_code))
    
    # Footer with language-specific help text
    st.markdown("""---""")
    st.markdown(f"""
        <div style="text-align: center; padding: 20px;">
            <p>{translate_text("Need help? Contact your local agricultural extension office", lang_code)}</p>
            <p style="color: #666;">{translate_text("Made with ❤️ for Smart Agriculture", lang_code)}</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()