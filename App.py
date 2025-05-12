import easyocr
import streamlit as st
import cv2
import numpy as np
from PIL import Image
from fuzzywuzzy import fuzz
import hashlib
import json
import os
import random
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import joblib

# MUST BE FIRST STREAMLIT COMMAND
st.set_page_config(page_title="AI Cyberbullying System", layout="centered")

USER_DATA_FILE = "user_data.json"
BANNED_USERS_FILE = "banned_users.json"

ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "admin123"

reader = easyocr.Reader(['en'])

@st.cache_resource
def load_models():
    text_model = joblib.load("cyberbullying_model_optimized.pkl")

    image_model = models.resnet50(pretrained=False)
    num_ftrs = image_model.fc.in_features

    saved_state_dict = torch.load("best_resnet50.pth", map_location=torch.device('cpu'))
    num_classes_saved = saved_state_dict['fc.weight'].shape[0]

    image_model.fc = torch.nn.Linear(num_ftrs, num_classes_saved)

    image_model.load_state_dict(saved_state_dict)

    image_model.eval()
    return text_model, image_model

text_model, image_model = load_models()

def analyze_text(text):
    try:
        prediction = text_model.predict([text])[0]
        labels = {0: "Not Bullying", 1: "Humorous", 2: "Personal", 3: "Severe"}
        return labels.get(prediction, "Unknown")
    except Exception as e:
        st.error(f"An error occurred during text analysis: {e}")
        return "Error during analysis"

def classify_comment(text):
    try:
        encoded_text = text.encode('utf-8', errors='ignore').decode('utf-8')
        prediction = text_model.predict([encoded_text])[0]
        return prediction
    except Exception:
        return 3

def predict_image_bullying(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = image_model(img_tensor)
        _, predicted = torch.max(output, 1)
        return predicted.item()

def load_user_data():
    if os.path.exists(USER_DATA_FILE):
        with open(USER_DATA_FILE, 'r') as file:
            return json.load(file)
    return {}

def save_user_data(user_data):
    with open(USER_DATA_FILE, 'w') as file:
        json.dump(user_data, file)

def load_banned_users():
    if os.path.exists(BANNED_USERS_FILE):
        with open(BANNED_USERS_FILE, 'r') as file:
            return json.load(file)
    return []

def save_banned_users(banned_users):
    with open(BANNED_USERS_FILE, 'w') as file:
        json.dump(banned_users, file)

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def create_profile(user_id, password):
    user_data = load_user_data()
    if user_id not in user_data:
        user_data[user_id] = {"password": hash_password(password)}
        save_user_data(user_data)
        return True
    return False

def authenticate_user(user_id, password):
    banned_users = load_banned_users()
    if user_id in banned_users:
        return False
    user_data = load_user_data()
    if user_id in user_data:
        if user_data[user_id]["password"] == hash_password(password):
            return True
    return False

def authenticate_admin(username, password):
    if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
        return True
    return False

def ban_user(user_id):
    banned_users = load_banned_users()
    if user_id not in banned_users:
        banned_users.append(user_id)
        save_banned_users(banned_users)
        return True
    return False

def load_bullying_keywords():
    keywords = []
    try:
        with open("bullying_keywords.txt", "r", encoding="utf-8") as file:
            for line in file:
                keyword = line.strip().lower()
                if keyword:
                    keywords.append(keyword)
    except FileNotFoundError:
        st.warning("bullying_keywords.txt not found. Creating empty list")
    return keywords

def load_non_bullying_keywords(file_path="nonbullying_keywords.txt"):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            keywords = [line.strip().lower() for line in file if line.strip()]
        return keywords
    except FileNotFoundError:
        st.warning("nonbullying_keywords.txt not found. Creating empty list.")
        return []

def is_non_bullying_text(text, nonbullying_keywords):
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in nonbullying_keywords)

def home_page():
    st.title("AI-Powered Cyberbullying System")
    image_folder = r"D:\Images"
    CAPTIONS = [
        "Such a beautiful view!",
        "Enjoying this moment!",
        "Nature at its best!",
        "Captured a perfect shot!",
        "Memories from today...",
        "A little bit of everything!",
        "This moment is everything!",
        "Can't stop smiling!"
    ]
    if os.path.exists(image_folder):
        image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if image_files:
            st.subheader("Posts")
            if 'post_states' not in st.session_state:
                st.session_state.post_states = {}
                st.session_state.image_captions = {}
                for i, image_file in enumerate(image_files):
                    st.session_state.post_states[i] = {'likes': 0, 'comments': []}
                    random_caption = random.choice(CAPTIONS)
                    st.session_state.image_captions[i] = random_caption
            for i, image_file in enumerate(image_files):
                image_path = os.path.join(image_folder, image_file)
                st.image(image_path, caption=st.session_state.image_captions[i], use_container_width=True)
                if st.button(f"👍 Like", key=f"like_button_{i}"):
                    st.session_state.post_states[i]['likes'] += 1
                st.write(f"Likes: {st.session_state.post_states[i]['likes']}")
                comment = st.text_input(f"💬 Add a comment:", key=f"comment_input_{i}")
                if st.button(f"📩 Comment on Post", key=f"comment_button_{i}"):
                    if comment:
                        non_bullying_keywords = load_non_bullying_keywords()
                        if is_non_bullying_text(comment, non_bullying_keywords):
                            st.session_state.post_states[i]['comments'].append(comment)
                            st.success("✅ Comment posted.")
                        else:
                            result = classify_comment(comment)
                            if result == 3:
                                st.error(f"⚠️ Comment contains **severe bullying** and cannot be posted.")
                            elif result in [1, 2]:
                                st.warning("⚠️ Your comment may contain inappropriate content. It will be sent for admin review.")
                            else:
                                st.session_state.post_states[i]['comments'].append(comment)
                                st.success("✅ Comment posted.")
                if st.session_state.post_states[i]['comments']:
                    st.write("📝 *Comments:*")
                    for c in st.session_state.post_states[i]['comments']:
                        st.write(f"- {c}")
                st.write("---")
        else:
            st.write("No images found in the folder.")
    else:
        st.error(f"The folder {image_folder} does not exist.")

def text_analysis():
    st.subheader("📝 Text Analysis")
    text = st.text_area("Enter text for analysis:")
    if st.button("Analyze Text"):
        if text.strip() == "":
            st.warning("Please enter some text to analyze.")
        else:
            non_bullying_keywords = load_non_bullying_keywords()
            if is_non_bullying_text(text, non_bullying_keywords):
                st.success("🟢 Skipped: Detected as harmless based on keyword match. This text is allowed.")
            else:
                result = analyze_text(text)
                st.info(f"🔍 Prediction: **{result}**")

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    denoised = cv2.fastNlMeansDenoising(thresh, None, 30, 7, 21)
    return denoised

def contains_bullying_text(text, bullying_keywords):
    text_lower = text.lower().split()
    exception_words = ["cute", "beautiful", "handsome"]
    for word in text_lower:
        if word in exception_words:
            continue
        for kw in bullying_keywords:
            if fuzz.partial_ratio(word, kw) > 90:
                return True
    return False

def image_analysis():
    st.subheader("🖼️ Image Analysis")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        prediction = predict_image_bullying(image)
        label_map = {0: "Not Bullying", 1: "Caution: Possible Bullying", 2: "Severe Bullying"}
        
        preprocessed_img = preprocess_image(image_np)
        st.write("🔍 Extracting text from image...")
        text_results = reader.readtext(preprocessed_img, detail=0)
        extracted_text = " ".join(text_results)
        if extracted_text:
            st.write(f"📝 Extracted Text: `{extracted_text}`")
            bullying_keywords = load_bullying_keywords()
            if contains_bullying_text(extracted_text, bullying_keywords):
                st.error("⚠️ This image contains **bullying content**!")
            else:
                st.success("✅ No harmful content detected.")
        else:
            st.info("ℹ️ No readable text found in the image.")

def admin_dashboard():
    st.subheader("📊 Admin Dashboard")
    user_data = load_user_data()
    banned_users = load_banned_users()
    total_users = len(user_data)
    total_banned = len(banned_users)
    total_active = max(0, total_users - total_banned)
    st.write(f"Number of Profiles: {total_users}")
    st.write(f"Number of Banned Users: {total_banned}")
    st.write(f"Total Active Users: {total_active}")
    st.subheader("👥 Registered Users")
    user_list = []
    for user in user_data.keys():
        status = "Banned" if user in banned_users else "Active"
        user_list.append({"User ID": user, "Status": status})
    st.table(user_list)
    st.subheader("📈 User Activity Statistics")
    if "post_states" in st.session_state:
        activity_data = []
        for post_id, post_info in st.session_state.post_states.items():
            activity_data.append({
                "Post ID": post_id,
                "Likes": post_info['likes'],
                "Comments": len(post_info['comments'])
            })
        st.table(activity_data)
    else:
        st.write("No user activity recorded yet.")
    st.subheader("📝 Recent Comments")
    if "post_states" in st.session_state:
        all_comments = []
        for post_id, post_info in st.session_state.post_states.items():
            for comment in post_info['comments']:
                all_comments.append(f"Post {post_id}: {comment}")
        if all_comments:
            for comment in all_comments[-5:]:
                st.write(f"💬 {comment}")
        else:
            st.write("No comments yet.")
    else:
        st.write("No comments available.")
    st.subheader("🔍 Search & Filter Users")
    search_query = st.text_input("Enter User ID to search:")
    if search_query:
        if search_query in user_data:
            status = "Banned" if search_query in banned_users else "Active"
            st.success(f"User {search_query} is **{status}**.")
        else:
            st.error("User not found.")
    labels = ['Banned Users', 'Active Users']
    sizes = [total_banned, total_active]
    colors = ['#FF6347', '#90EE90']
    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90, shadow=True,
           textprops={'fontsize': 14})
    ax.axis('equal')
    st.pyplot(fig)
    st.subheader("🚫 Ban/Unban User")
    user_id_to_ban = st.text_input("Enter User ID to Ban/Unban:")
    if st.button("Ban User"):
        if ban_user(user_id_to_ban):
            st.success(f"User {user_id_to_ban} has been banned.")
            st.rerun()
        else:
            st.error("User already banned.")
    if st.button("Unban User"):
        banned_users = load_banned_users()
        if user_id_to_ban in banned_users:
            banned_users.remove(user_id_to_ban)
            save_banned_users(banned_users)
            st.success(f"User {user_id_to_ban} has been unbanned.")
            st.rerun()
        else:
            st.error("User is not banned.")
    st.subheader("❌ Banned Users")
    if banned_users:
        for banned_user in banned_users:
            st.write(f"User ID: {banned_user}")
        else:
            st.write("No banned users.")

def main():
    if 'user_id' not in st.session_state and 'admin_logged_in' not in st.session_state:
        st.title("Sign In, Create Account, or Admin Login")
        option = st.radio("Select an option", ["Sign In", "Create Account", "Admin Login"])
        if option == "Sign In":
            st.subheader("Sign In")
            user_id_signin = st.text_input("Enter your User ID (Sign In):")
            password_signin = st.text_input("Enter your Password (Sign In):", type="password")
            if st.button("Sign In"):
                if authenticate_user(user_id_signin, password_signin):
                    st.session_state.user_id = user_id_signin
                    st.success(f"Logged in as {user_id_signin}")
                    st.rerun()
                else:
                    banned_users = load_banned_users()
                    if user_id_signin in banned_users:
                        st.error("Your account is banned. Please contact support.")
                    else:
                        st.error("Invalid credentials.")
        elif option == "Create Account":
            st.subheader("Create Account")
            user_id_signup = st.text_input("Enter your User ID (Create Account):")
            password_signup = st.text_input("Enter your Password (Create Account):", type="password")
            if st.button("Create Account"):
                if create_profile(user_id_signup, password_signup):
                    st.success(f"Account created for {user_id_signup}")
                else:
                    st.error("Account already exists.")
        elif option == "Admin Login":
            st.subheader("Admin Login")
            admin_username = st.text_input("Enter Admin Username:")
            admin_password = st.text_input("Enter Admin Password:", type="password")
            if st.button("Login as Admin"):
                if authenticate_admin(admin_username, admin_password):
                    st.session_state.admin_logged_in = True
                    st.success(f"Logged in as Admin")
                    st.rerun()
                else:
                    st.error("Invalid admin credentials.")
    if 'user_id' in st.session_state or 'admin_logged_in' in st.session_state:
        if st.sidebar.button("Logout"):
            if 'admin_logged_in' in st.session_state:
                del st.session_state['admin_logged_in']
            if 'user_id' in st.session_state:
                del st.session_state['user_id']
            st.success("Logged out successfully.")
            st.rerun()
    if 'admin_logged_in' in st.session_state and st.session_state.admin_logged_in:
        page = st.sidebar.selectbox("Choose a page", ["Home", "Text Analysis", "Image Analysis", "Admin Dashboard"])
        if page == "Home":
            home_page()
        elif page == "Text Analysis":
            text_analysis()
        elif page == "Image Analysis":
            image_analysis()
        elif page == "Admin Dashboard":
            admin_dashboard()
    elif 'user_id' in st.session_state:
        page = st.sidebar.selectbox("Choose a page", ["Home", "Text Analysis", "Image Analysis"])
        if page == "Home":
            home_page()
        elif page == "Text Analysis":
            text_analysis()
        elif page == "Image Analysis":
            image_analysis()

if __name__ == "__main__":
    main()