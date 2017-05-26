from keras.models import model_from_json


def load_model(path, name):
    json_file = open(path + name + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(path + name + '.h5')
    loaded_model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
    return loaded_model
