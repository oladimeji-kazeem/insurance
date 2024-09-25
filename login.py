import streamlit as st
import sqlite3
import hashlib

# Set the page configuration (only needs to be done once, at the top of the script)
st.set_page_config(page_title="Login Page", layout="wide")

# Database setup
conn = sqlite3.connect('users.db')
c = conn.cursor()

# Create users table without verification columns
c.execute('''CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY,
                first_name TEXT NOT NULL,
                last_name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE,
                password TEXT NOT NULL)''')
conn.commit()

# Helper functions
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(stored_password, provided_password):
    return stored_password == hash_password(provided_password)

def get_user_by_email(email):
    """Get user by email."""
    c.execute('SELECT * FROM users WHERE email=?', (email,))
    return c.fetchone()

# Registration function
def register_user(first_name, last_name, email, password):
    c.execute('INSERT INTO users (first_name, last_name, email, password) VALUES (?, ?, ?, ?)', 
              (first_name, last_name, email, hash_password(password)))
    conn.commit()

# Registration page
def register():
    st.title("User Registration")

    first_name = st.text_input("First Name")
    last_name = st.text_input("Last Name")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    confirm_password = st.text_input("Confirm Password", type="password")
    captcha = st.checkbox("I am a human being")

    if st.button("Register"):
        if not captcha:
            st.warning("Please confirm you are a human.")
        elif password != confirm_password:
            st.warning("Passwords do not match.")
        else:
            if get_user_by_email(email):
                st.warning("Email is already registered.")
            else:
                register_user(first_name, last_name, email, password)
                st.success("You have successfully registered! You can now log in.")

# Login page
def login():
    st.title("User Login")

    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        user = get_user_by_email(email)
        if user:
            if verify_password(user[4], password):
                st.success(f"Welcome {user[1]} {user[2]}!")
                st.session_state['logged_in'] = True
                st.session_state['email'] = email
                st.session_state['first_name'] = user[1]
                st.session_state['last_name'] = user[2]

                # Set session state to navigate to the main menu
                st.session_state['current_page'] = "menu"
            else:
                st.error("Invalid email or password.")
        else:
            st.error("Email not registered.")

# Menu-based navigation after login
def main_menu():
    st.title("Main Menu")
    st.write(f"Hello, {st.session_state['first_name']} {st.session_state['last_name']}!")
    
    # Menu options
    menu = st.selectbox("Choose an option:", ["Credit Risk App", "Logout"])

    # Handle menu navigation
    if menu == "Credit Risk App":
        # Navigate to the Credit Risk App
        st.session_state['current_page'] = "credit_risk_app"
    elif menu == "Logout":
        # Handle logout by resetting the session state
        st.session_state['logged_in'] = False
        st.session_state['current_page'] = "login"

# Main app flow
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

if 'current_page' not in st.session_state:
    st.session_state['current_page'] = "login"  # Default to login page

# Determine which page to show based on session state
if st.session_state['logged_in']:
    if st.session_state['current_page'] == "menu":
        main_menu()
    elif st.session_state['current_page'] == "credit_risk_app":
        st.title("Credit Risk Prediction App")
        st.write(f"Welcome to the Credit Risk App, {st.session_state['first_name']} {st.session_state['last_name']}!")
        # Placeholder for Credit Risk App functionality
        if st.button("Go back to menu"):
            st.session_state['current_page'] = "menu"
        if st.button("Logout"):
            # Handle logout
            st.session_state['logged_in'] = False
            st.session_state['current_page'] = "login"
else:
    # Show login or registration page when not logged in
    st.title("Welcome! Please log in or register.")
    option = st.selectbox("Choose an option", ("Login", "Register"))
    if option == "Login":
        login()
    elif option == "Register":
        register()
