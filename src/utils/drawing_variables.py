def pe6_colormap():
    """Returns a list of colors for PE6 variables.

    Returns:
        list: A list of color codes for PE6 variables.
    """
    # TODO: add key-value pairs for each color

    color_list = [
        "#a0a0a4",  # PEmax
        "#55a0fb",  # PEmaxdRNaseH
        "#f9ccf0",  # PE6a
        "#fff0a0",  # PE6b
        "#04be78",  # PE6c
        "#8d8dff",  # PE6d
        "#c0b57b",  # PE6e
        "#ffa0a0",  # PE6f
        "#f96495",  # PE6g
    ]

    return color_list


def pe6_colormap_dict() -> dict:
    """Returns a dictionary containing color codes for different variables in the PE6 colormap.

    Returns:
        dict: A dictionary mapping variable names to their corresponding color codes.
    """

    # To use it in palette argument
    color_dict = {
        "PEmax": "#a0a0a4",
        "PEmaxdRNaseH": "#55a0fb",
        "PE6c(+PEmaxCas9)": "#f9ccf0",
        "PE6d(+PEmaxCas9)": "#fff0a0",
        "PE6a(+PEmaxCas9)": "#04be78",
        "PE6g(+dRNaseH)": "#8d8dff",
        "PE6b(+PEmaxCas9)": "#c0b57b",
        "PE6f(+dRNaseH)": "#ffa0a0",
        "PE6e(+dRNaseH)": "#f96495",
    }

    return color_dict
