def get_dict(names: list, values: list, display: bool) -> dict:
    """
    Creates a dictionary by pairing corresponding elements from two input iterables.

    Args:
        names (list): An iterable containing keys for the dictionary.
        values (list): An iterable containing values for the dictionary.
        display (bool): A flag to indicate whether to display the dictionary. Defaults to True.

    Returns:
        dict: The dictionary created from pairing the elements of 'names' and 'values'.

    Example:
        names = ["loss", "acc"]
        values = [0.5, 0.4]
        result = get_dict(names, values, display=True)
        # Output: loss: 0.5 , acc: 0.4 ,

    """
    result = zip(names, values)
    dictionary = {k: v for k, v in result}
    if display:
        for key, value in zip(dictionary.keys(), dictionary.values()):
            print(key, ":", f"{value: .4f}", ", ", end="")
    return dictionary
