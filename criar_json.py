import json

words_to_keep = ['The Eiffel Tower', 'ambulance', 'barn', 'cactus', 'dolphin', 
                 'floor lamp', 'harp', 'moon', 'penguin', 'sink']


with open("words_to_keep.json", "w") as js:
    json.dump(words_to_keep, js)
