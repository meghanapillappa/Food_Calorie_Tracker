import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import tempfile
import json
import os
from datetime import datetime

# ---------------------------------------------------
# Nutrition Database
# ---------------------------------------------------
NUTRITION = {
    "banana": {"calories": 105},
    "apple": {"calories": 95},
    "orange": {"calories": 62},
    "lemon": {"calories": 17},
    "lime": {"calories": 20},
    "pineapple": {"calories": 50},
    "mango": {"calories": 99},
    "papaya": {"calories": 55},
    "grapes": {"calories": 62},
    "egg": {"calories": 78},
    "bread": {"calories": 66},
    "idli": {"calories": 39},
    "dosa": {"calories": 133},

    # Added Indian Foods
    "biryani": {"calories": 290},          # 1 cup (rice + chicken mix)
    "veg_biryani": {"calories": 240},
    "chicken_biryani": {"calories": 300},

    "dal": {"calories": 120},              # 1 cup cooked dal
    "sambar": {"calories": 100},           # 1 cup

    "chapati": {"calories": 120},          # 1 medium
    "rice": {"calories": 136},             # 1 cup cooked
    "curd": {"calories": 98},              # 1 cup

    "pongal": {"calories": 210},           # 1 bowl
    "poha": {"calories": 180},             # 1 cup
    "upma": {"calories": 220},             # 1 cup

    "vada": {"calories": 97},              # 1 medu vada
    "samosa": {"calories": 262},           # 1 piece

    "mysore_pak": {"calories": 390},       # 1 piece
    "gulab_jamun": {"calories": 150},      # 1 ball
    "laddu": {"calories": 186},            # 1 laddu

    "paneer": {"calories": 265},           # 100g
    "chole": {"calories": 210},            # 1 cup
    "rajma": {"calories": 230},            # 1 cup
}


MEAL_TYPES = ["Breakfast", "Lunch", "Dinner", "Snacks"]

# ---------------------------------------------------
# Load YOLO Model
# ---------------------------------------------------
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# ---------------------------------------------------
# Load or Create Meal Log
# ---------------------------------------------------
LOG_FILE = "meal_log.json"

if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w") as f:
        json.dump({}, f)

def load_log():
    with open(LOG_FILE, "r") as f:
        return json.load(f)

def save_log(data):
    with open(LOG_FILE, "w") as f:
        json.dump(data, f, indent=4)


# ---------------------------------------------------
# PAGE UI
# ---------------------------------------------------
st.title("🍽️ Smart Food Detector + Meal Tracker + Calorie Advisor")

tab_detect, tab_tracker, tab_profile = st.tabs([
    "📸 Detect Food",
    "📘 Meal Tracker",
    "👤 Profile & Advice"
])

# ---------------------------------------------------
# PROFILE TAB
# ---------------------------------------------------
with tab_profile:
    st.header("👤 User Profile")

    sex = st.selectbox("Sex", ["Male", "Female"])
    age = st.number_input("Age", 10, 100, 25, step=1, format="%d")
    weight = st.number_input("Weight (kg)", 20, 300, 70, step=1, format="%d")
    height = st.number_input("Height (cm)", 120, 230, 170, step=1, format="%d")

    activity = st.selectbox("Activity Level", [ 
        "Sedentary",
        "Lightly Active",
        "Moderately Active",
        "Very Active",
        "Extra Active"
    ])

    goal = st.selectbox("Goal", ["Maintain", "Lose Weight", "Gain Weight"])

    if st.button("Calculate Daily Recommended Calories"):
        # BMR
        if sex == "Male":
            bmr = 10 * weight + 6.25 * height - 5 * age + 5
        else:
            bmr = 10 * weight + 6.25 * height - 5 * age - 161

        # Activity multiplier
        mult = {
            "Sedentary": 1.2,
            "Lightly Active": 1.375,
            "Moderately Active": 1.55,
            "Very Active": 1.725,
            "Extra Active": 1.9,
        }[activity]

        tdee = bmr * mult

        # Goal adjustment
        if goal == "Lose Weight":
            recommended = tdee - 500
        elif goal == "Gain Weight":
            recommended = tdee + 500
        else:
            recommended = tdee

        recommended = int(recommended)

        st.success(f"🎯 Daily Recommended Calories: **{recommended} kcal**")
        st.session_state["recommended"] = recommended


# ---------------------------------------------------
# DETECTION TAB
# ---------------------------------------------------
with tab_detect:
    st.header("📸 Upload Food Image for Detection")

    uploaded = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        img_np = np.array(img)
        st.image(img, width=500)

        # Save temp file
        temp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        img.save(temp.name)

        # Detect
        results = model(temp.name)[0]

        st.subheader("🔍 Detected Items (Editable)")

        final_items = []

        for i, box in enumerate(results.boxes):
            cls = int(box.cls[0])
            detected_name = results.names[cls].lower()

            # If YOLO detects unknown → default apple
            if detected_name not in NUTRITION:
                detected_name = "apple"

            # Editable item name with UNIQUE KEY
            name = st.selectbox(
                f"Detected item {i+1}",
                list(NUTRITION.keys()),
                index=list(NUTRITION.keys()).index(detected_name),
                key=f"label_{i}"
            )

            # Editable count with UNIQUE KEY + FIXED TYPE
            count = st.number_input(
                f"Count for {name}",
                min_value=1,
                max_value=20,
                value=1,
                step=1,
                key=f"count_{i}",
                format="%d"
            )

            final_items.append((name, count))

        st.write("---")

        # Choose meal
        meal_choice = st.selectbox("Add calories to:", MEAL_TYPES)

        if st.button("➕ Add to Meal Log"):
            today = datetime.now().strftime("%Y-%m-%d")
            log = load_log()

            if today not in log:
                log[today] = {m: 0 for m in MEAL_TYPES}

            total_plate = sum(NUTRITION[name]["calories"] * count for name, count in final_items)

            log[today][meal_choice] += total_plate
            save_log(log)

            st.success(f"Added **{total_plate} kcal** to **{meal_choice}**")


# ---------------------------------------------------
# MEAL TRACKER TAB
# ---------------------------------------------------
with tab_tracker:
    st.header("📘 Today’s Meal Summary")

    today = datetime.now().strftime("%Y-%m-%d")
    log = load_log()

    if today not in log:
        st.info("No meals logged today yet.")
    else:
        meals = log[today]
        total_today = sum(meals.values())

        st.write("### 🍽️ Calories by Meal")
        for m in MEAL_TYPES:
            st.write(f"**{m}:** {meals[m]} kcal")

        st.write("---")
        st.write(f"### 🔥 Total Today: **{total_today} kcal**")

        # Compare with recommended calories
        if "recommended" in st.session_state:
            target = st.session_state["recommended"]

            if total_today < target - 150:
                st.success("🟢 UNDER target — good for weight loss.")
            elif total_today <= target + 150:
                st.info("🟡 WITHIN target — healthy range.")
            else:
                st.error("🔴 ABOVE target — adjust tomorrow’s meals.")
