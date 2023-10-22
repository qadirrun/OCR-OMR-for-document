import streamlit as st
from PIL import Image
from OCR_main import process_image

st.title("Image Processing and Data Extraction")

# Allow users to upload an image
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    st.write("Processing the uploaded image...")

    # Convert the PIL.Image.Image object to a file and save it to a local path
    image_path = "uploaded_image.jpg"  # Choose a suitable file name and extension

    # Open the image and convert it to RGB mode
    image = Image.open(uploaded_image)
    image = image.convert("RGB")
    image.save(image_path)

    # Display the uploaded image
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Process the uploaded image using your function and pass the file path
    first_name, first_traveling_date, first_flight_number, controlled_item_indicator,output_plot_file = process_image(image_path)

    # Display the extracted results
    st.write("Name:", first_name)
    st.write("Traveling Date:", first_traveling_date)
    st.write("Flight Number(s):", first_flight_number)
    st.write("Controlled Item(s) Indicator:", controlled_item_indicator)

    st.image(output_plot_file, caption='Output Plot', use_column_width=True)

    # Save the extracted data to a local text file
    with open("extracted_data.txt", "w") as file:
        file.write(f"Name: {first_name}\n")
        file.write(f"Traveling Date: {first_traveling_date}\n")
        file.write(f"Flight Number(s): {first_flight_number}\n")
        file.write(f"Controlled Item(s) Indicator: {controlled_item_indicator}\n")

    st.write("Data has been processed and extracted. Extracted data has been saved to 'extracted_data.txt'.")


