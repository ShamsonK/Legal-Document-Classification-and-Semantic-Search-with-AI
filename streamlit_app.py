import streamlit as st
import pandas as pd
from main_app import process_data_and_train_models, perform_semantic_search

# Streamlit App
st.title('Advanced AI Models for Legal Text Classification and Query Matching')

# File uploader
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # Read the data
    data = pd.read_csv(uploaded_file)
    df = pd.DataFrame(data)

    # Ensure 'Raw Content' and 'Offense' columns exist
    if 'Raw Content' not in df.columns or 'Offense' not in df.columns:
        st.error("The CSV file must contain 'Raw Content' and 'Offense' columns.")
    else:
        # Process data and train models
        results = process_data_and_train_models(df)

        # Display evaluation results
        st.header('Text Classification')
        st.subheader('Random Forest')
        st.write(f"Accuracy: {results['accuracy_rf']}")
        st.write(f"Precision: {results['precision_rf']}")
        st.write(f"Recall: {results['recall_rf']}")
        st.write(f"F1 Score: {results['f1_rf']}")

        st.subheader('XGBoost')
        st.write(f"Accuracy: {results['accuracy_xgb']}")
        st.write(f"Precision: {results['precision_xgb']}")
        st.write(f"Recall: {results['recall_xgb']}")
        st.write(f"F1 Score: {results['f1_xgb']}")

        # Semantic Search
        st.header('Semantic Search')
        query = st.text_input('Enter your query:')
        if query:
            best_matches = perform_semantic_search(df, query)
            st.write(f"Top Results: {best_matches}")
