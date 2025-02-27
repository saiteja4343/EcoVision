import streamlit as st

def process_image(img, model, confidence, class_filter=None, co2_data=None):
    try:
        results = model.predict(
            source=img,
            conf=confidence,
            classes=[co2_data[co2_data['Foodstuff'] == c].index[0] for c in class_filter] if class_filter else None,
            imgsz=640,
            device=model.device.type
        )
        plotted = results[0].plot()[:, :, ::-1]
        return results[0], plotted
    except Exception as e:
        st.error(f"Processing error: {str(e)}")
        return None, None
