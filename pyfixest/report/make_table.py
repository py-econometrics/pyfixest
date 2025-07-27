import re
from typing import Optional

import pandas as pd
from great_tables import GT


def make_table(
    df: pd.DataFrame,
    type: str = "gt",
    notes: str = "",
    rgroup_sep: str = "tb",
    rgroup_display: bool = True,
    caption: Optional[str] = None,
    tab_label: Optional[str] = None,
    texlocation: str = "htbp",
    full_width: bool = False,
    file_name: Optional[str] = None,
    **kwargs,
):
    r"""
    Create a booktab style table in the desired format (gt or tex) from a DataFrame.
    The DataFrame can have a multiindex. Column index used to generate horizonal
    table spanners. Row index used to generate row group names and
    row names. The table can have multiple index levels in columns and up to
    two levels in rows.


    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the table to be displayed.
    type : str, optional
        Type of table to be created. The default is 'gt'.
    notes : str
        Table notes to be displayed at the bottom of the table.
    rgroup_sep : str
        Whether group names are separated by lines. The default is "tb".
        When output type = 'gt', the options are 'tb', 't', 'b', or '', i.e.
        you can specify whether to have a line above, below, both or none.
        When output type = 'tex' no line will be added between the row groups
        when rgroup_sep is '' and otherwise a line before the group name will be added.
    rgroup_display : bool
        Whether to display row group names. The default is
        True.
    caption : str
        Table caption to be displayed at the top of the table. The default is None.
        When either caption or label is provided the table will be wrapped in a
        table environment.
    tab_label : str
        LaTex label of the table. The default is None. When either caption or label
        is provided the table will be wrapped in a table environment.
    texlocation : str
        Location of the table. The default is 'htbp'.
    full_width : bool
        Whether to expand the table to the full width of the page. The default is False.
    file_name : str
        Name of the file to save the table to. The default is None.
        gt tables will be saved as html files and latex tables as tex files.

    Returns
    -------
    A table in the specified format.
    """
    assert isinstance(df, pd.DataFrame), "df must be a pandas DataFrame."
    assert not isinstance(df.index, pd.MultiIndex) or df.index.nlevels <= 2, (
        "Row index can have at most two levels."
    )
    assert type in ["gt", "tex"], "type must be either 'gt' or 'tex'."
    assert rgroup_sep in [
        "tb",
        "t",
        "b",
        "",
    ], "rgroup_sep must be either 'tb', 't', 'b', or ''."
    assert file_name is None or (
        isinstance(file_name, str) and file_name.endswith((".html", ".tex"))
    ), "file_name must end with '.html' or '.tex'."

    # Make a copy of the DataFrame to avoid modifying the original
    dfs = df.copy()

    # Produce LaTeX code if either type is 'tex' or the
    # user has passed a file_name which ends with '.tex'
    if type == "tex" or (isinstance(file_name, str) and file_name.endswith(".tex")):
        # First wrap all cells which contain a line break in a makecell command
        dfs = dfs.map(
            lambda x: f"\\makecell{{{x}}}" if isinstance(x, str) and "\\\\" in x else x
        )
        row_levels = dfs.index.nlevels
        # when the row index has more than one level, we will store
        # the top level to use later to add clines and row group titles
        # and then remove it
        if row_levels > 1:
            # Store the top level of the row index
            top_row_id = dfs.index.get_level_values(0).to_list()
            # Generate a list of the distinct values
            row_groups = list(dict.fromkeys(top_row_id))
            # Generate a list containing the number of rows for each group
            row_groups_len = [top_row_id.count(group) for group in row_groups]
            # Drop the top level of the row index:
            dfs.index = dfs.index.droplevel(0)

        # Style the table
        styler = dfs.style
        # if caption is not None:
        #     styler.set_caption(caption)

        # Generate LaTeX code
        latex_res = styler.to_latex(
            hrules=True,
            multicol_align="c",
            multirow_align="t",
            column_format="l" + "c" * (dfs.shape[1] + dfs.index.nlevels),
        )

        # # Now perform post-processing of the LaTeX code
        # # First split the LaTeX code into lines
        lines = latex_res.splitlines()
        # Find the line number of the \midrule
        line_at = next(i for i, line in enumerate(lines) if "\\midrule" in line)
        # Add space after this \midrule:
        lines.insert(line_at + 1, "\\addlinespace")
        line_at += 1

        # When there are row groups then insert midrules and groupname
        if row_levels > 1 and len(row_groups) > 1:
            # Insert a midrule after each row group
            for i in range(len(row_groups)):
                if rgroup_display:
                    # Insert a line with the row group name & same space around it
                    # lines.insert(line_at+1, "\\addlinespace")
                    lines.insert(line_at + 1, "\\emph{" + row_groups[i] + "} \\\\")
                    lines.insert(line_at + 2, "\\addlinespace")
                    lines.insert(line_at + 3 + row_groups_len[i], "\\addlinespace")
                    line_at += 3
                if (rgroup_sep != "") and (i < len(row_groups) - 1):
                    # For tex output we only either at a line between the row groups or not
                    # And we don't add a line after the last row group
                    line_at += row_groups_len[i] + 1
                    lines.insert(line_at, "\\midrule")
                    lines.insert(line_at + 1, "\\addlinespace")
                    line_at += 1
        else:
            # Add line space before the end of the table
            lines.insert(line_at + dfs.shape[0] + 1, "\\addlinespace")

        # Insert cmidrules (equivalent to column spanners in gt)
        # First find the first line with an occurrence of "multicolumn"
        cmidrule_line_number = None
        for i, line in enumerate(lines):
            if "multicolumn" in line:
                cmidrule_line_number = i + 1
                # Regular expression to find \multicolumn{number}
                pattern = r"\\multicolumn\{(\d+)\}"
                # Find all matches (i.e. values of d) in the LaTeX string & convert to integers
                ncols = [int(match) for match in re.findall(pattern, line)]

                cmidrule_string = ""
                leftcol = 2
                for n in ncols:
                    cmidrule_string += (
                        r"\cmidrule(lr){"
                        + str(leftcol)
                        + "-"
                        + str(leftcol + n - 1)
                        + "} "
                    )
                    leftcol += n
                lines.insert(cmidrule_line_number, cmidrule_string)

        # # Put the lines back together
        latex_res = "\n".join(lines)

        # Wrap in threeparttable to allow for table notes
        if notes is not None:
            latex_res = (
                "\\begin{threeparttable}\n"
                + latex_res
                + "\n\\footnotesize "
                + notes
                + "\n\\end{threeparttable}"
            )
        else:
            latex_res = (
                "\\begin{threeparttable}\n" + latex_res + "\n\\end{threeparttable}"
            )

        # If caption or label specified then wrap in table environment
        if (caption is not None) or (tab_label is not None):
            latex_res = (
                "\\begin{table}["
                + texlocation
                + "]\n"
                + "\\centering\n"
                + ("\\caption{" + caption + "}\n" if caption is not None else "")
                + ("\\label{" + tab_label + "}\n" if tab_label is not None else "")
                + latex_res
                + "\n\\end{table}"
            )

        # Set cell aligment to top
        latex_res = "\\renewcommand\\cellalign{t}\n" + latex_res

        # Set table width to full page width if full_width is True
        # This is done by changing the tabular environment to tabular*
        if full_width:
            latex_res = latex_res.replace(
                "\\begin{tabular}{l", "\\begin{tabularx}{\\linewidth}{X"
            )
            latex_res = latex_res.replace(
                "\\end{tabular}", "\\end{tabularx}\n \\vspace{3pt}"
            )
            # with tabular*
            # latex_res = latex_res.replace("\\begin{tabular}{", "\\begin{tabular*}{\linewidth}{@{\extracolsep{\\fill}}")
            # latex_res = latex_res.replace("\\end{tabular}", "\\end{tabular*}")

        if file_name is not None:
            with open(file_name, "w") as f:
                f.write(latex_res)  # Write the latex code to a file

        if type == "tex":
            return latex_res

    if type == "gt":
        # GT does not support MultiIndex columns, so we need to flatten the columns
        if isinstance(dfs.columns, pd.MultiIndex):
            # Store labels of the last level of the column index (to use as column names)
            col_names = dfs.columns.get_level_values(-1)
            nl = dfs.columns.nlevels
            # As GT does not accept non-unique column names: so to allow for them
            # we just assign column numbers to the lowest index level
            col_numbers = list(map(str, range(len(dfs.columns))))
            # Save the whole column index in order to generate table spanner labels later
            dfcols = dfs.columns.to_list()
            # Then flatten the column index just numbering the columns
            dfs.columns = pd.Index(col_numbers)
            # Store the mapping of column numbers to column names
            col_dict = dict(zip(col_numbers, col_names))
            # Modify the last elements in each tuple in dfcols
            dfcols = [(t[:-1] + (col_numbers[i],)) for i, t in enumerate(dfcols)]
            # And drop the first column as we don't want table spanners on top of the variables
            # WE DON'T NEED THIS WITH ROW INDEX dfcols = dfcols[1:]
        else:
            nl = 1

        rowindex = dfs.index

        # Now reset row index to have the index as columns to be displayed in the table
        dfs.reset_index(inplace=True)

        # And specify the rowname_col and groupname_col
        if isinstance(rowindex, pd.MultiIndex):
            rowname_col = dfs.columns[1]
            groupname_col = dfs.columns[0]
        else:
            rowname_col = dfs.columns[0]
            groupname_col = None

        # Generate the table with GT
        gt = GT(dfs, auto_align=False)

        # When caption is provided, add it to the table
        if caption is not None:
            gt = (
                gt.tab_header(title=caption).tab_options(
                    table_border_top_style="hidden",
                )  # Otherwise line above caption
            )

        if nl > 1:
            # Add column spanners based on multiindex
            # Do this for every level in the multiindex (except the one with the column numbers)
            for i in range(nl - 1):
                col_spanners: dict[str, list[str | int]] = {}
                # Iterate over columns and group them by the labels in the respective level
                for c in dfcols:
                    key = c[i]
                    if key not in col_spanners:
                        col_spanners[key] = []
                    col_spanners[key].append(c[-1])
                for label, columns in col_spanners.items():
                    gt = gt.tab_spanner(label=label, columns=columns, level=nl - 1 - i)
            # Restore column names
            gt = gt.cols_label(**col_dict)

        # Customize the table layout
        gt = (
            gt.tab_source_note(notes)
            .tab_stub(rowname_col=rowname_col, groupname_col=groupname_col)
            .tab_options(
                table_border_bottom_style="hidden",
                stub_border_style="hidden",
                column_labels_border_top_style="solid",
                column_labels_border_top_color="black",
                column_labels_border_bottom_style="solid",
                column_labels_border_bottom_color="black",
                column_labels_border_bottom_width="0.5px",
                column_labels_vlines_color="white",
                column_labels_vlines_width="0px",
                table_body_border_top_style="solid",
                table_body_border_top_width="0.5px",
                table_body_border_top_color="black",
                table_body_hlines_style="none",
                table_body_vlines_color="white",
                table_body_vlines_width="0px",
                table_body_border_bottom_color="black",
                row_group_border_top_style="solid",
                row_group_border_top_width="0.5px",
                row_group_border_top_color="black",
                row_group_border_bottom_style="solid",
                row_group_border_bottom_width="0.5px",
                row_group_border_bottom_color="black",
                row_group_border_left_color="white",
                row_group_border_right_color="white",
                data_row_padding="4px",
                column_labels_padding="4px",
            )
            .cols_align(align="center")
        )

        # Full page width
        if full_width:
            gt = gt.tab_options(table_width="100%")

        # Customize row group display
        if "t" not in rgroup_sep:
            gt = gt.tab_options(row_group_border_top_style="none")
        if "b" not in rgroup_sep:
            gt = gt.tab_options(row_group_border_bottom_style="none")
        if not rgroup_display:
            gt = gt.tab_options(
                row_group_font_size="0px",
                row_group_padding="0px",
            )
        # Save the html code of the table to a file
        if file_name is not None:
            with open(file_name, "w") as f:
                f.write(gt.as_raw_html())

        return gt
