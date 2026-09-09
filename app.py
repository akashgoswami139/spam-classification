import streamlit as st
from pipeline import predict_message


st.set_page_config(
    page_title="SMS Spam Detector",
    page_icon="📱",
    layout="centered",
)

st.title("📱 SMS Spam Detector")
st.write("Enter an SMS message and the trained Linear SVM will classify it as Ham or Spam.")

message = st.text_area(
    "Enter your message",
    placeholder="Example: Congratulations! You have won a free prize!",
    height=180,
)

if st.button("Predict", type="primary", use_container_width=True):
    if not message.strip():
        st.warning("Please enter a message.")
    else:
        try:
            result = predict_message(message)

            if result["prediction"] == 1:
                st.error(f"🚨 {result['label']}")
            else:
                st.success(f"✅ {result['label']}")

            st.caption(result["explanation"])
            st.write(f"Decision score: `{result['decision_score']:.4f}`")

        except FileNotFoundError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
