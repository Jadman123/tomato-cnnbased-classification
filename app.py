import streamlit as st
import pandas as pd
import tensorflow as tf
import numpy as np
import io
from PIL import Image
import sqlite3
import time
import matplotlib.pyplot as plt
import os

# 🔹 Step 1: Configure Streamlit Page (must be the first command)
st.set_page_config(page_title="Tomato Disease Classification", layout="centered")

# Enable WAL mode for better concurrency handling
def enable_wal_mode():
    conn = sqlite3.connect("predictions.db")
    cursor = conn.cursor()
    cursor.execute("PRAGMA journal_mode=WAL;")  # ✅ Enable WAL mode
    conn.commit()
    conn.close()

enable_wal_mode()

# Load model only once
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("C:/code/tomato-deseases/saved_models/jadman.keras")

model = load_model()

# Class labels and organic treatment recommendations
class_names = ["Tomato_Early_blight", "Tomato_Late_blight", "Tomato_Leaf_Mold", "Tomato_healthy", "Unknown"]

recomendation = {
    "Tomato_Early_blight": {
        "high": "🌿 Apply neem oil (natural antifungal) or compost tea to infected leaves. Use a baking soda spray (1 tsp baking soda + 1L water). Remove and dispose of infected leaves immediately.",
        "medium": "🌱 Apply organic copper-based sprays sparingly. Prune excess leaves to improve airflow and reduce moisture.",
        "low": "🛑 Suspected Early Blight. Monitor closely, remove yellowing leaves, and boost soil health with compost."
    },
    "Tomato_Late_blight": {
        "high": "🍂 Use a garlic spray (crushed garlic + water) or potassium bicarbonate spray to slow down fungal spread. Remove heavily infected plants immediately to prevent further contamination.",
        "medium": "🌿 Apply a milk spray (1 part milk, 2 parts water) to create a protective barrier. Avoid overhead watering and keep plants dry.",
        "low": "🤔 Suspected Late Blight. Improve airflow, prune lower leaves, and monitor for signs of rapid spread."
    },
    "Tomato_Leaf_Mold": {
        "high": "🌱 Apply chamomile tea spray as a natural antifungal. Increase ventilation and use sulfur-based organic treatments if needed.",
        "medium": "🌿 Remove affected leaves and mist plants with diluted apple cider vinegar (1 tbsp per liter of water).",
        "low": "⚠️ Possible Leaf Mold. Keep leaves dry, space plants apart, and monitor closely for changes."
    },
    "Tomato_healthy": {
        "high": "✅ Your plant is healthy! Maintain good watering habits, enrich soil with compost, and rotate crops regularly.",
        "medium": "🧐 Looks healthy, but monitor for early disease symptoms. Consider using compost tea for added immunity.",
        "low": "🤔 Uncertain, but appears healthy. Keep soil well-drained and avoid overwatering."
    },
    "Unknown": {
        "high": "🔍 The disease is not recognized. Consider consulting an expert or researching further.",
        "medium": "🤷 Uncertain. Try rotating crops, boosting soil health, and applying organic preventive sprays.",
        "low": "🧐 Not sure what this is. Monitor plant health closely and check for new symptoms in a few days."
    }
}

# Database connection with retry mechanism
def execute_query(query, params=()):
    retries = 5
    for i in range(retries):
        try:
            conn = sqlite3.connect("predictions.db", timeout=10)  # ✅ Add timeout for locking issues
            cursor = conn.cursor()
            cursor.execute(query, params)
            result = cursor.fetchall()
            conn.commit()
            conn.close()
            return result
        except sqlite3.OperationalError as e:
            if "database is locked" in str(e):
                time.sleep(0.5)  # ✅ Wait and retry
            else:
                raise e  # Raise other errors immediately

# Fetch user credentials
def get_user_from_db(username, password):
    # Check if user is an Admin
    query_admin = "SELECT admin_id, farm_id FROM admin_table WHERE username=? AND password=?"
    result_admin = execute_query(query_admin, (username, password))
    
    if result_admin:
        return {"role": "admin", "user_id": result_admin[0][0], "farm_id": result_admin[0][1]}
    
    # Check if user is a Farmer
    query_farmer = "SELECT farmer_id, farm_id FROM farmer_table WHERE username=? AND password=?"
    result_farmer = execute_query(query_farmer, (username, password))
    
    if result_farmer:
        return {"role": "farmer", "user_id": result_farmer[0][0], "farm_id": result_farmer[0][1]}
    
    return None  # If no match is found


# Fetch predictions for the farm
def get_predictions_for_farm(farm_id):
    query = """
        SELECT p.prediction_id, f.name, p.predicted_class, p.confidence, p.date, p.recomendation, p.image 
        FROM predicted_table p 
        JOIN farmer_table f ON p.farmer_id = f.farmer_id 
        WHERE p.farm_id=?
    """
    data = execute_query(query, (farm_id,))
    df = pd.DataFrame(data, columns=["ID", "Farmer Name", "Prediction", "Confidence", "Date", "Recommendation", "Image"])
    # ✅ Ensure Confidence is a Float and Format as Percentage
    df["Confidence"] = df["Confidence"].astype(float)  # Convert to float
    df["Confidence"] = df["Confidence"].apply(lambda x: f"{x:.2f}%")  # Format to 'xx.xx%'

    return df  


# Fetch history for the farm
def get_history_for_farm(farm_id):
    query = """
        SELECT h.history_id, h.prediction_id, f.name, h.predicted_class, 
               (h.confidence || '%') AS confidence,  -- ✅ Maglalagay ng % symbol
               h.date, h.timestamp, h.action, h.recomendation, h.image
        FROM history_table h
        JOIN farmer_table f ON h.farmer_id = f.farmer_id
        WHERE h.farm_id=?
    """
    data = execute_query(query, (farm_id,))
    return pd.DataFrame(data, columns=[
        "History ID", "Prediction ID", "Farmer Name", "Predicted Class", "Confidence Level", 
        "Date", "Timestamp", "Action", "Recommendation", "Image"
    ])  

# Add farmer to database
def add_farmer_to_db(farmer_name, farmer_username, farmer_password, contact_number, farm_id):
    query = """
        INSERT INTO farmer_table (name, username, password, contact_number, farm_id) 
        VALUES (?, ?, ?, ?, ?)
    """
    execute_query(query, (farmer_name, farmer_username, farmer_password, contact_number, farm_id))
    
def check_existing_username(username):
    query = "SELECT COUNT(*) FROM farmer_table WHERE username = ?"
    result = execute_query(query, (username,))
    return result[0][0] > 0  # Returns True if username exists, otherwise False

def delete_expired_predictions():
    conn = sqlite3.connect("predictions.db", check_same_thread=False)
    cursor = conn.cursor()
    
    cursor.execute("""
        DELETE FROM predicted_table
        WHERE (strftime('%s', 'now', '+8 hours') - strftime('%s', date)) >= 172800
    """)
    conn.commit()  # ✅ Para mase-save ang changes
    conn.close()

# Authentication
if "authentication_status" not in st.session_state:
    st.session_state.authentication_status = False
    st.session_state.role = None  # Track whether user is admin or farmer

if not st.session_state.authentication_status:
    # 🍅 Header sa taas ng box
    st.markdown("<h2 style='text-align: center; margin-top: 60px; color: #1fd5e4;'>🍅 Tomato Disease Detection</h2>", unsafe_allow_html=True)  
    with st.form("login_form"):
        username = st.text_input("Username", key="username")
        password = st.text_input("Password", type="password", key="password")
        submit_button = st.form_submit_button("Login")
    if submit_button:
        user = get_user_from_db(username, password)  # ✅ Function to validate login
        if user:
            st.session_state.authentication_status = True
            st.session_state.user_id = user["user_id"]
            st.session_state.farm_id = user["farm_id"]
            st.session_state.role = user["role"]
            # ✅ Initialize farm_name before querying the database
            st.session_state.farm_name = "Unknown Farm"

            # ✅ Fetch farm name using farm_id
            conn = sqlite3.connect("predictions.db")
            cursor = conn.cursor()
    
            cursor.execute("SELECT farm_name FROM farm_table WHERE farm_id = ?", (st.session_state.farm_id,))
            farm = cursor.fetchone()
    
            if farm:
                st.session_state.farm_name = farm[0]  # ✅ Store farm name
            else:
                st.session_state.farm_name = "Unknown Farm"  # ❌ Default if not found
    
            conn.close()

            if user["role"] == "admin":
                st.success(f"✅ Welcome, Admin {username}!")
            else:
                st.success(f"✅ Welcome, Farmer {username} of {st.session_state.farm_name}!")  # Display correct farm

        else:
            st.error("❌ Incorrect Username or Password.")

else:
      # Admin Dashboard
    if st.session_state.role == "admin":
        st.sidebar.title("Admin Dashboard")
        st.markdown("<div style='text-align: center;'><h4>Welcome ADMIN!</h4></div>", unsafe_allow_html=True)
        st.header("🍅 Tomato Disease Detection")
    
        # Tabs for different sections
        tab1, tab2, tab3 = st.tabs(["Predicted Table", "History Table", "Add Farmer"])

        # Predicted Table
        with tab1:
            st.header("📊 Predicted Table")
            delete_expired_predictions()
            predicted_data = get_predictions_for_farm(st.session_state.farm_id)
            st.dataframe(predicted_data, use_container_width=True)

        # History Table
        with tab2:
            st.header("📜 History Table")
            history_data = get_history_for_farm(st.session_state.farm_id)

            if history_data.empty:
                st.warning("⚠️ No history records found.")
            else:
                st.dataframe(history_data, use_container_width=True)

        # Add Farmer Tab
        with tab3:
            st.header("👨‍🌾 Add Farmer")

            # ✅ Create a separate dictionary for form inputs
            farmer_data = {
                "farmer_name": "",
                "farmer_username": "",
                "farmer_password": "",
                "contact_number": ""
            }

            with st.form("add_farmer_form", clear_on_submit=True):  # ✅ clear_on_submit automatically resets the form
                farmer_name = st.text_input("Farmer Name", placeholder="Enter the farmer's name")
                farmer_username = st.text_input("Username", placeholder="Enter a unique username")
                farmer_password = st.text_input("Password", placeholder="Enter a secure password", type="password")
                contact_number = st.text_input("Contact Number", placeholder="Enter farmer's contact number")
        
                submitted = st.form_submit_button("➕ Add Farmer")

                if submitted:
                    if farmer_name and farmer_username and farmer_password and contact_number:
                        existing_user = check_existing_username(farmer_username)  # ✅ Check if username already exists
                        if existing_user:
                            st.error("⚠️ Username already exists! Please choose a different one.")
                        else:
                            add_farmer_to_db(farmer_name, farmer_username, farmer_password, contact_number, st.session_state.farm_id)
                            st.success(f"✅ Farmer '{farmer_name}' added successfully!")
                    else:
                        st.error("⚠️ All fields are required!")
        

        # Farmer content (if logged in as user/farmer)
    elif st.session_state.role == "farmer":
        
        conn = sqlite3.connect("predictions.db", check_same_thread=False)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM farmer_table WHERE farmer_id=?", (st.session_state.user_id,))
        farmer_name = cursor.fetchone()
        conn.close()

        # 🔹 Extract name from DB result
        farmer_name = farmer_name[0] if farmer_name else "Farmer"

        # 🔹 Update Sidebar Title with dynamic name
        st.sidebar.markdown(f"""
            <h3 style='margin-bottom: -15px; margin-top: 70px;'>👨‍🌾 Hi, {farmer_name}!</h3>
            <h1 style='margin-top: 0px;'>Farmer Dashboard</h1>
        """, unsafe_allow_html=True)

        # Farmer UI for Home and About
        with st.sidebar.expander("☰ Menu", expanded=True):
            app_mode = st.radio("Select Page", ["Overview", "About", "Disease Detection", "⚙️ Settings"])

        if app_mode == "⚙️ Settings":
            st.markdown(f"<h2 style='text-align: center;'> ⚙️ Settings</h2>", unsafe_allow_html=True)
            settings_tab = st.radio("Settings", ["Account Management"])
        
            # ✅ Use st.form() to enable automatic clearing
            with st.form("update_account_form", clear_on_submit=True):
                new_name = st.text_input("Name", placeholder="Enter new name")
                new_username = st.text_input("Username", placeholder="Enter new username")
                new_password = st.text_input("Password", placeholder="Enter new password", type="password")
                new_contact = st.text_input("Contact Number", placeholder="Enter new contact number")
                submitted = st.form_submit_button("Update Account")

                if submitted:
                    conn = sqlite3.connect("predictions.db", check_same_thread=False)
                    cursor = conn.cursor()
                    
                    # Fetch current data
                    cursor.execute("SELECT name, username, password, contact_number FROM farmer_table WHERE farmer_id=?", (st.session_state.user_id,))
                    current_data = cursor.fetchone()
                    
                    # Keep old values if new inputs are empty
                    updated_name = new_name if new_name else current_data[0]
                    updated_username = new_username if new_username else current_data[1]
                    updated_password = new_password if new_password else current_data[2]
                    updated_contact = new_contact if new_contact else current_data[3]
                    
                    # Update only changed fields
                    cursor.execute("""
                        UPDATE farmer_table SET name=?, username=?, password=?, contact_number=? WHERE farmer_id=?
                    """, (updated_name, updated_username, updated_password, updated_contact, st.session_state.user_id))
                    
                    conn.commit()
                    conn.close()
                    
                    st.success("Account updated successfully!")
                    # ✅ Refresh UI
                    st.rerun()
        elif app_mode == "Overview": 
            # 🔹 Centered Welcome Message
            st.markdown(f"<h2 style='text-align: center;'>👩‍🌾 {st.session_state.farm_name}!</h2>", unsafe_allow_html=True)
            st.markdown("""
                ### 🌿 Welcome to the Tomato Disease Detection System!  

                This system helps farmers **detect tomato plant diseases** in real-time using **image-based AI detection**. Simply **take or upload an image** of a tomato leaf, and the system will analyze it to determine if the plant has a disease.  

                ### 🔍 How to Use the System:  
                1. **Take a photo** of the tomato leaf or **upload an image** from your device.  
                2. The system will **analyze the image** using a trained deep learning model.  
                3. If a disease is detected, you will receive **organic treatment recommendations** to help manage the issue.  
                4. View **disease statistics and model accuracy** to monitor farm health.  

                ### 🏆 What This System Offers:  
                ✅ **Fast & Accurate Disease Detection** – Uses AI to identify diseases in just seconds.  
                ✅ **Organic Treatment Suggestions** – Provides eco-friendly solutions to maintain healthy crops.  
                ✅ **Farm-Specific Dashboard** – Each farm can see their own reports and statistics.  
                ✅ **Real-Time Insights** – View the latest disease trends and model accuracy.  

                ### 🔬 How This System Was Built:  
                - **Deep Learning Model (CNN)** – Trained on thousands of tomato leaf images.  
                - **Image Processing** – Used resizing, normalization, and enhancement techniques.  
                - **Streamlit** – For easy and interactive web-based visualization.  
                - **SQLite Database** – Stores predictions and user information.  

                This tool empowers farmers by **reducing crop losses and improving disease management** through AI-driven insights! 🌱🚜  
                """, unsafe_allow_html=True)
            conn = sqlite3.connect("predictions.db", check_same_thread=False)
            overview_tab, history_tab = st.tabs(["📊 Overview", "📜 History"])
            with overview_tab:
                col1, col2 = st.columns(2)
                st.markdown(f"#####   ")
                with col1:
                    st.subheader("📈 Disease Detection Overview")
                    st.write("This chart shows the number of disease detections recorded in the last 10 days.")
                    query = f"""
                    SELECT date, predicted_class 
                    FROM predicted_table 
                    WHERE farm_id = '{st.session_state.farm_id}'
                    ORDER BY date DESC 
                    LIMIT 10
                    """
                    df = pd.read_sql_query(query, conn)
        
                    if not df.empty:
                        df["date"] = pd.to_datetime(df["date"])
                        disease_counts = df.groupby("date").count()
                        st.line_chart(disease_counts)
                    else:
                        st.info("No disease detection records found for this farm.")

                with col2:
                    st.subheader("📊 Disease Prediction Statistics")
                    st.write("This bar chart shows the most frequently detected diseases in your farm.")
                    query = f"""
                    SELECT predicted_class, COUNT(*) as count 
                    FROM predicted_table 
                    WHERE farm_id = '{st.session_state.farm_id}'
                    GROUP BY predicted_class
                    """
                    df = pd.read_sql_query(query, conn)
        
                    if not df.empty:
                        disease_stats = df.set_index("predicted_class")["count"].to_dict()
                        st.bar_chart(disease_stats)
                    else:
                        st.info("No disease prediction data available.")
                        # 🔹 Disease Prediction Statistics (Bar Chart)
                        st.subheader("📊 Disease Prediction Statistics")
                        st.write("This bar chart shows the most frequently detected diseases in your farm. The higher the bar, the more common the disease.")

                        query = f"""
                        SELECT predicted_class, COUNT(*) as count 
                        FROM predicted_table 
                        WHERE farm_id = '{st.session_state.farm_id}'
                        GROUP BY predicted_class
                        """
                        df = pd.read_sql_query(query, conn)

                        if not df.empty:
                            disease_stats = df.set_index("predicted_class")["count"].to_dict()
                            st.bar_chart(disease_stats)
                        else:
                            st.info("No disease prediction data available.")
            with history_tab:
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("📈 History Detection Overview")
                    st.write("This chart shows the number of disease detections recorded in the last 10 days (Historical Data).")
                    query = f"""
                    SELECT date, predicted_class 
                    FROM history_table 
                    WHERE farm_id = '{st.session_state.farm_id}'
                    ORDER BY date DESC 
                    LIMIT 10
                    """
                    df_history = pd.read_sql_query(query, conn)
        
                    if not df_history.empty:
                        df_history["date"] = pd.to_datetime(df_history["date"])
                        history_counts = df_history.groupby("date").count()
                        st.line_chart(history_counts)
                    else:
                        st.info("No historical disease detection records found for this farm.")

                with col2:
                    st.subheader("📊 History Prediction Statistics")
                    st.write("This bar chart shows the most frequently detected diseases in your farm based on historical data.")
                    query = f"""
                    SELECT predicted_class, COUNT(*) as count 
                    FROM history_table 
                    WHERE farm_id = '{st.session_state.farm_id}'
                    GROUP BY predicted_class
                    """
                    df_history_stats = pd.read_sql_query(query, conn)
        
                    if not df_history_stats.empty:
                        history_stats = df_history_stats.set_index("predicted_class")["count"].to_dict()
                        st.bar_chart(history_stats)
                    else:
                        st.info("No historical disease prediction data available.")
                
            # 🔹 Model Accuracy Calculation (Without is_correct)
            st.subheader("🎯 Model Accuracy")
            st.write("This shows the overall accuracy of the disease detection model based on recorded predictions.")
            model_accuracy = 85.4  # Replace this with your actual model accuracy
            st.metric("Overall Model Accuracy", f"{model_accuracy:.2f}%")
            st.markdown(f"#####   ")
            # 🔹 Additional Visualization: Pie Chart for Disease Distribution
            st.subheader("🍕 Disease Distribution")
            st.write("This pie chart represents the proportion of different diseases detected in your farm. This helps in understanding which diseases are most dominant.")

            if not df.empty:
                fig, ax = plt.subplots()
                ax.pie(disease_stats.values(), labels=disease_stats.keys(), autopct="%1.1f%%", startangle=90)
                ax.axis("equal")  # Equal aspect ratio ensures the pie chart is circular
                st.pyplot(fig)
            else:
                st.info("No data available for disease distribution.")

            conn.close()


        elif app_mode == "About":
            st.header("🌱 About the Developers")
            st.markdown(
                """
                Welcome to the **Tomato Disease Detection System**! 🌿
        
                This system is designed to help farmers identify tomato plant diseases using AI-based image classification. 
                It provides **real-time detection** and **organic treatment recommendations**, empowering farmers to take proactive steps in protecting their crops.
        
                The project was developed with passion and dedication by our team. Meet the developers behind this innovation!
                """
            )
    
            # Developer Profile Card
            def display_developer(name, image, address, phone, email, motto):
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.image(image, width=230)
                with col2:
                    st.subheader(name)
                    st.markdown("**📍 Address:** " + address)
                    st.markdown("**📞 Phone:** " + phone)
                    st.markdown("**📧 Email:** " + email)
                    st.markdown("**🏆 Motto:** \"" + motto + "\"")
                st.markdown("---")
    
            # Example Developers (Replace with real data)
            st.markdown(f"#####   ")
            display_developer(
                name="Allen Kim S. Jadman - Lead Developer",
                image="allen.jpeg",
                address="Brgy. Kay-Anlog, Calamba, Laguna",
                phone="+63 907 546 7562",
                email="jadmanallenkim@gmail.com",
                motto="Don’t be afraid of challenges; debug your fears, refine your relationships, and embrace happiness with an open heart."
            )
            st.markdown(f"#####   ")
            display_developer(
                name="Laarne R. Florida - Documentation",
                image="laarne.jpeg",
                address="Blk 73 lot 3 Golden City, Santa Rosa Laguna",
                phone="+63 967 758 0971",
                email="laarniflorida6@gmail.com",
                motto="Live with purpose, love with passion."
            )
            st.markdown(f"#####   ")
            display_developer(
                name="Rayna Salve A. Estremera - Documentation",
                image="estre.jpeg",
                address="Malitlit, Sta. Rosa Laguna",
                phone="+63 918 604 2290",
                email="raynasalveestremera543@gmail.com",
                motto="Don't Stop When you're Tired, Stop When you're Done."
            )

        elif app_mode == "Disease Detection":
            st.header("🍅 Tomato Disease Detection")
            # Farmer-specific UI for Disease Detection
            tab1 = st.radio("Select Action", ["Upload Image", "Take a Picture"])
            image = None

            if tab1 == "Upload Image":
                test_image = st.file_uploader("Choose an Image:", type=["jpg", "jpeg", "png"])
                if test_image:
                    image = Image.open(test_image)

            elif tab1 == "Take a Picture":
                camera_image = st.camera_input("Take a picture")
                if camera_image:
                    image = Image.open(camera_image)
                    image_name = "captured_image.jpg" 
            def predict_with_unknown_class(image, model, threshold=0.7):
                preds = model.predict(image)
                class_idx = np.argmax(preds)
                confidence = preds[0][class_idx] * 100
                return ("Unknown", confidence) if confidence < (threshold * 100) else (class_names[class_idx], confidence)

            if image:
                st.image(image, caption="Uploaded Image", use_column_width=True)
                
                # ✅ Convert image to model-compatible format (numpy array)
                image_resized = image.convert("RGB").resize((224, 224))
                img_array = np.array(image_resized) / 255.0
                img_array = tf.expand_dims(img_array, 0)
                
                predicted_class, confidence = predict_with_unknown_class(img_array, model)
                confidence_level = "high" if confidence >= 0.9 else "medium" if confidence >= 0.8 else "low"
                
                st.write(f"### 🏷️ Prediction: **{predicted_class}**")
                st.write(f"### 🎯 Confidence: **{confidence:.2f}%**")
                st.write(f"     ")
                # ✅ Fixed Reference to "recomendation"
                if predicted_class in recomendation and confidence_level in recomendation[predicted_class]:
                    recomendation_text = recomendation[predicted_class][confidence_level]  # ⚫ Fixed reference
                else:
                    recomendation_text = "No recommendation available."

                st.write(f"##### 🌿 Recommendation: {recomendation_text}")
                st.write(f"     ")

                # Save Prediction Button
                if st.button("Save Prediction"):
                    try:
                        conn = sqlite3.connect("predictions.db", check_same_thread=False)
                        cursor = conn.cursor()

                        # Ensure user_id exists
                        farmer_id = st.session_state.user_id if "user_id" in st.session_state else None
                        farm_id = st.session_state.farm_id

                        filename = f"tomato_{int(time.time())}.png"
                        img_path = os.path.join("images", filename)  # Path to save the image

                        # Ensure the directory exists
                        os.makedirs("images", exist_ok=True)

                        # Save the image as a file instead of binary
                        image.save(img_path, format="PNG")
                        recomendation_text = recomendation[predicted_class][confidence_level]  # ⚫ Corrected variable name


                        # Insert prediction into database
                        cursor.execute("""
                            INSERT INTO predicted_table (farm_id, predicted_class, confidence, image, farmer_id, recomendation)
                            VALUES (?, ?, ?, ?, ?, ?)
                        """, (farm_id, predicted_class, confidence, filename, farmer_id, recomendation_text))

                        conn.commit()
                        conn.close()

                        st.success("✅ Prediction and image saved successfully! 📥")

                    except Exception as e:
                        st.error(f"❌ Error saving prediction: {e}")

    # 🔹 Step 6: Logout Button (for both admin and farmer)
    if st.sidebar.button("Log-out"):
        st.session_state.authentication_status = False
        st.session_state.user_name = ""
        st.session_state.clear()  # Clear session state and reload page