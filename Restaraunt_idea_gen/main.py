import streamlit as st
import langchain_helper

st.title("Restaurant Idea Generator")

cuisine = st.sidebar.selectbox("Pick a cuisine", ["Italian", "Chinese", "Mexican", "Indian", "Thai"])


if cuisine:
    response = langchain_helper.generate_restaurant_idea_and_items(cuisine)

    st.header(response['restaurant_name'].strip())
    menu_items = response['menu_items'].strip().split(',')

    st.write("### Menu Items")
    for item in menu_items:
        st.write(f"- {item.strip()}")

