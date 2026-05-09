def predict_waste_image(image_path, model, class_names):
    img = tf.keras.utils.load_img(image_path, target_size=IMG_SIZE)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array, verbose=0)
    predicted_index = np.argmax(predictions[0])
    confidence = predictions[0][predicted_index]

    predicted_class = class_names[predicted_index]

    plt.figure(figsize=(5, 5))
    plt.imshow(img)
    plt.axis("off")
    plt.title(f"Prediction: {predicted_class}\nConfidence: {confidence:.2%}")
    plt.show()

    return predicted_class, confidence

# Example:
image_path = "/content/dataset/Garbage classification/Garbage classification/metal/metal102.jpg"
predicted_class, confidence = predict_waste_image(image_path, model, class_names)
print(predicted_class, confidence)
