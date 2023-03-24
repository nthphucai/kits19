def get_dict(names, values, display=False) -> dict:
    result = zip(names, values)
    dictionary = {k: v for k, v in result}
    if display:
        for key, value in zip(dictionary.keys(), dictionary.values()):
            print(key, ":", f"{value: .4f}", ", ", end="")
    return dictionary
