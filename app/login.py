import streamlit as st

def login():
    users = {
        "admin": "1234",
        "user1": "abcd"
    }

    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False

    if not st.session_state["authenticated"]:
        st.title("IQVIA Login")

        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            login_btn = st.form_submit_button("Login")

            if login_btn:
                if username in users and users[username] == password:
                    st.session_state["authenticated"] = True
                    st.session_state["username"] = username
                    st.success(f"Welcome, {username}!")
                    st.rerun()
                else:
                    st.error("Invalid username or password")
        st.stop()