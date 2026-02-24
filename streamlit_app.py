import streamlit as st

file = open("report.txt", "r", encoding="utf-8")
st.write(file.read())
