pred = effnet.predict(img)
pred_rotated_45 = effnet.predict(img_rotated_45)
pred_rotated_90 = effnet.predict(img_rotated_90)
pred_darkened = effnet.predict(img_darkened)
pred_brightened = effnet.predict(img_brightened)
decoded_original = tensorflow.keras.applications.imagenet_utils.decode_predictions(pred)
decoded_rotated_45 = tensorflow.keras.applications.imagenet_utils.decode_predictions(pred_rotated_45)
decoded_rotated_90 = tensorflow.keras.applications.imagenet_utils.decode_predictions(pred_rotated_90)
decoded_darkened = tensorflow.keras.applications.imagenet_utils.decode_predictions(pred_darkened)
decoded_brightened = tensorflow.keras.applications.imagenet_utils.decode_predictions(pred_brightened)