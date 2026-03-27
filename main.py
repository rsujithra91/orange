import streamlit as st
st.title("Interactive streamlit App")
name=st.text_input("Enter your name")
if st.button("submit"):
  st.write("Hello,{name}! Welcome to streamlit")
