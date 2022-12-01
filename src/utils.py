"""Utility functions that are generically useful."""

def collect(iterable):
    """Stores each element of an iterable object into a list.

    Args:
        iterable: An object to be looped over to return elements.

    Returns:
        A list storing the elements from looping over iterable.
    """
    storage = []
    for element in iterable:
        storage.append(element)
    return storage

def collect_to_array(storage, iterable):
    """Stores each element of an iterable object to an array.

    Assumes that the iterable outputs a tuple in the form
    (index, element), where the index is used to decide where to store
    the element in the storage array.

    Args:
        storage: An array which is mutated to store the elements from
            iterable.
        iterable: An object which can be looped over to return
            (index, element) tuples.

    Returns:
        A reference to the storage array.
    """
    for index, element in iterable:
        storage[index] = element
    return storage
