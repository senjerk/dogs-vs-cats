import tensorflow as tf
import cv2

loaded_model = tf.keras.models.load_model("cats_vs_dogs.model")

def predict_image(image_path):
    IMG_SIZE = 70
    img_array = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    prepared_image = new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    prepared_image = prepared_image / 255.0

    prediction = loaded_model.predict(prepared_image)

    if prediction > 0.5:
        print(f"The image is of a Dog with {round(prediction[0][0] * 100, 2)}% confidence.")
    else:
        print(f"The image is of a Cat with {round((1 - prediction[0][0]) * 100, 2)}% confidence.")


image_path = "path_to_test_image.jpg"
predict_image("cat.jpg")
