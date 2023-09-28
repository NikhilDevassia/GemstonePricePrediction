import sys

import streamlit as st

from Gemstone.config import CustomException
from Gemstone.config import logging
from Gemstone.pipeline import CustomData, PredictPipeline

# Set the Streamlit configuration to use port 8080
st.set_config(backend="server")
st.set_option("server.port", 8501)

st.set_page_config(
    page_title="Gemstone",
    page_icon="👨🏻‍💻",
    layout="wide",
    initial_sidebar_state="expanded",
)

def gemstone_price_prediction(carat, depth, table, x, y, z, cut, color, clarity):
    try:
        data = CustomData(
            carat=float(carat),
            depth=float(depth),
            table=float(table),
            x=float(x),
            y=float(y),
            z=float(z),
            cut=cut,
            color=color,
            clarity=clarity
        )
        df = data.get_data_as_data_frame()
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(df)
        logging.info(f'Prediction Completed and the result is : {results}')
        return results
    except Exception as e:
        logging.info(
            "Exited the gemstone_price_prediction method prediction goes wrong!")
        raise CustomException(e, sys) from e


def load_screen_items():
    st.markdown("<h1 style='text-align: center;'> 💎 Gemstone Price 💸 Prediction</h1>",
                unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    gemstone_price = ""
    with col1:
        carat = st.text_input("Carat")
    with col2:
        cut = st.selectbox('Cut',
                           ('Fair', 'Good', 'Very Good', 'Premium', 'Ideal'))
    with col3:
        color = st.selectbox('Color',
                             ('D', 'E', 'F', 'G', 'H', 'I', 'J'))
    with col1:
        clarity = st.selectbox('Clarity',
                               ('I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF'))
    with col2:
        depth = st.text_input("Depth")
    with col3:
        table = st.text_input("Table")
    with col1:
        x = st.text_input("x")
    with col2:
        y = st.text_input("y")
    with col3:
        z = st.text_input("z")

    if st.button("Process"):
        gemstone_price = gemstone_price_prediction(
            carat, depth, table, x, y, z, cut, color, clarity)

    if gemstone_price != "":
        st.success(f"Predicted Result : {round(gemstone_price[0])}")


if __name__ == "__main__":
    load_screen_items()
