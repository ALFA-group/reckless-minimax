"""
Python module for handling LaTeX processing
"""


def df_2_tex(df, filepath):
    """
    writes a df to tex file
    :param df: dataframe to be converted into tex table
    :param filepath: tex filepath
    :return:
    """
    tex_prefix = r"""\documentclass{standalone}
    \usepackage{booktabs}
    \begin{document}"""

    tex_suffix = r"""\end{document}"""

    with open(filepath, "w") as f:
        f.write(tex_prefix)
        f.write(df.to_latex(float_format="%.8f"))
        f.write(tex_suffix)



