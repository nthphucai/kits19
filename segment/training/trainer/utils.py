def get_dict(names, values, verbose=False) -> dict:
    result = zip(names, values)
    dictionary = {k: v for k, v in result}
    if verbose:
        for key, value in zip(dictionary.keys(), dictionary.values()):
            print(key, ":", f"{value: .4f}", ", ", end="")
    return dictionary
