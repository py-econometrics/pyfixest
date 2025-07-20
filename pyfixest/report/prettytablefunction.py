# Gelbach 2016 Decomposition Functions
# Written by Erica Ryan (@emryan) 6/19/2025

'''
This script contains functions to implement the decomposition proposed in Gelbach 2016.
There are two main scripts, the first is to run the original decomposition by variables
The second allows you to run a decomposition by groups of variables. For example, if
you wanted to see how much of an effect occupation could explain, you might want to
include several occupation indicators and then sum the effect. That would utilize
the grouped function.
'''

###########################################################################################################
# Preliminaries

# When writing this script, the available version of pyfixest was 0.24.2 (running pip install -U didn't help)
# The decompose function was added with 0.27, so we are going to install from github directly
# The decompose function in the github repo does not have functionality to have multiple categories for
# x1 or multiple x1 variables, so I forked it and made a version that does. Until the main package is
# updated (currently talking to the package owner about a push request), I'll pull from my github

#This is the version on my personal github
pip install git+https://github.com/Erica-Ryan/pyfixest.git

#This is the version on gitfarm. I wasn't able to pip install directly from it because of a midway auth issue
#https://code.amazon.com/packages/STK-prod-qoh-crest-ghs-sere/trees/mainline/--/emryan/gelbach2016/pyfixest

import pyfixest as pft
import pandas as pd
import numpy as np
from scipy import stats
from patsy import Treatment
import re
import numexpr
pd.options.mode.chained_assignment = None
np.random.seed(42)


###########################################################################################################
# Gelbach Decomposition
"""
Performs a statistical decomposition analysis to understand differences between groups and their contributing factors.

Parameters:
----------
data : pandas.DataFrame
    The input dataset containing all required variables.
y : str
    The dependent variable name in the dataset.
x1_vars : list
    List of baseline variables, including the decomposition variable. Must contain at least one variable.
    Note: All non-decomposition factor/categorical variables should be specified using C( )
x2_dict : dict
    Dictionary mapping display names to variable names for decomposition factors.
    Format: {'Display Name': 'variable_name', ...}
    Note: All factor/categorical variables should be specified using C( )
decomp_var_dict : dict
    Single-item dictionary with display name and variable name for the decomposition variable.
    Format: {'Display Name': 'variable_name'}
    Note: You do NOT have to specify C( ). It automatically assumes for the decomposition variable.
ref_cat : str
    Reference category for the decomposition variable.
decimals : int
    Number of decimal places in the output.
groups : bool or dict
    If False, individual decomposition. If dict, groups variables for combined effects.
    Format: {'Group Name': ['var1', 'var2'], ...}
percents : bool, optional (default=True)
    Whether to include percentage contributions in output.
stars : bool, optional (default=True)
    Whether to display significance stars (*p<0.1, **p<0.05, ***p<0.01).
conf_ints : bool, optional (default=False)
    Whether to display confidence intervals.
pvals : bool, optional (default=True)
    Whether to display p-values.

Returns:
-------
pandas.DataFrame
    Decomposition results table with estimates, confidence intervals (if requested),
    p-values (if requested), and percentage contributions (if requested).
"""

def run_decomposition(data, y, x1_vars, x2_dict, decomp_var_dict, ref_cat, decimals, groups, percents = True, stars = True, conf_ints = False, pvals = True, decomp_cat_mapping = False):
    x2_vars = list(x2_dict.values())
    decomp_var_nicename = list(decomp_var_dict.keys())[0]
    decomp_var = list(decomp_var_dict.values())[0]

    # Get the base decomp_var (without C() if it exists)
    decomp_var_base = decomp_var.split('(')[1].split(')')[0] if decomp_var.startswith('C(') else decomp_var

    # Check if decomp_var exists in x1_vars in either form
    decomp_var_in_x1 = False
    for x1_var in x1_vars:
        x1_var_base = x1_var.split('(')[1].split(')')[0] if x1_var.startswith('C(') else x1_var
        if x1_var_base == decomp_var_base:
            decomp_var_in_x1 = True
            x1_var_to_remove = x1_var  # Store the actual form found in x1_vars
            break

    if not x1_vars:
        raise ValueError("x1_vars cannot be empty - decomposition requires at least a decomposition variable. Hint: Did you include your decomposition variable in x1_vars?")

    if not decomp_var_in_x1:
        raise ValueError(f"Decomposition variable {decomp_var_base} must be included in x1_vars")

    if not x2_vars:
        raise ValueError("x2_vars cannot be empty - decomposition requires variables to decompose into")

    # Create x1_string excluding the exact form of decomp_var found in x1_vars
    x1_string = ' + '.join([var for var in x1_vars if var != x1_var_to_remove])

    x2_string = ' + '.join(x2_vars)

    # Create the full formula
    if isinstance(ref_cat, (int, float)):
        print('NUMERIC')
        formula = f"{y} ~ C({decomp_var_base}, Treatment(reference={ref_cat}))"
    else:
        print('STRING')
        formula = f"{y} ~ C({decomp_var_base}, Treatment(reference='{ref_cat}'))"

    # Only add x1_string if it's not empty
    if x1_string:
        formula += f" + {x1_string}"

    # Add x2_string if it exists
    if x2_string:
        formula += f" + {x2_string}"

    # Run the regression
    model = pft.feols(formula, data=data)

    #model_table = pft.make_table(model)

    # Get the values our decomposition variable takes, and the other x1 vars
    if isinstance(ref_cat, (int, float)):
        decomp_coefs = [coef for coef in model.coef().index.tolist() if f"C({decomp_var_base}, Treatment(reference={ref_cat}))[T." in coef]

    else:
        decomp_coefs = [coef for coef in model.coef().index.tolist() if f"C({decomp_var_base}, Treatment(reference='{ref_cat}'))[T." in coef]

    other_x1_vars = [var for var in x1_vars
                     if (var.split('(')[1].split(')')[0] if var.startswith('C(') else var) != decomp_var_base]

    # For each value of the decomp_var, we run the decompose function and save the results
    # Note, we control for all the x1 vars, and all other values of decomp_var

    def clean_tab(tab, stars=stars, pvals=pvals, conf_ints=conf_ints, percents=percents):
        tab.index.name = 'var'
        tab.reset_index(inplace=True)
        tab['type'] = np.where(tab['var'] == '', 'conf_int', 'val')
        tab['var'] = tab['var'].replace('', np.nan).ffill()

        tab1 = tab.head(2)
        tab2 = tab.tail(len(tab)-2).drop(['direct_effect', 'full_effect'], axis=1)

        tables_and_effects = [
            (tab1, ['direct_effect', 'full_effect', 'explained_effect']),
            (tab2, ['explained_effect'])
        ]


        for i, (table, effects) in enumerate(tables_and_effects):
            for effect in effects:
                table[[f'{effect}_lower', f'{effect}_upper']] = (
                    table[effect].shift(-1)
                    .str.strip('[]')
                    .str.split(',', expand=True)
                    .astype(float)
                )

                mask = table['type'] != 'conf_int'

                table.loc[mask, f'{effect}_pval'] = 2 * (1 - stats.norm.cdf(
                    abs(table.loc[mask, effect].astype(float) /
                        ((table.loc[mask, f'{effect}_upper'] - table.loc[mask, f'{effect}_lower']) /
                         (2 * stats.norm.ppf(0.975))))
                ))

                if stars:
                    table.loc[mask, effect] = table.loc[mask, effect].astype(str) + table.loc[mask, f'{effect}_pval'].apply(
                        lambda x: '***' if x <= 0.01 else
                                 '**' if 0.01 < x <= 0.05 else
                                 '*' if 0.05 < x <= 0.1 else
                                 '')

            # After processing all effects, clean up the columns
            columns_to_drop = [col for col in table.columns if
                              col.endswith(('_upper', '_lower')) or
                              col.startswith(('explained_effect_upper', 'explained_effect_lower'))]
            table.drop(columns=columns_to_drop, inplace=True)  # Use inplace=True


            # Handle p-values
            if pvals:
                tab_1 = table.loc[:, [col for col in table.columns if not col.endswith('pval')]]
                tab_2 = table.loc[:, ['var', 'type'] + [col for col in table.columns if col.endswith('pval')]]
                tab_2 = tab_2[tab_2['type'] != 'conf_int']

                tab_2['type'] = 'p_val'
                tab_2.columns = [col.replace('_pval', '') if col != 'vars' and col != 'type' else col for col in tab_2.columns]

                # Update the original table in tables_and_effects
                if i == 0:
                    tab1 = pd.concat([tab_1, tab_2], ignore_index=True, axis=0)
                else:
                    tab2 = pd.concat([tab_1, tab_2], ignore_index=True, axis=0)
            else:
                # Drop p-value columns if not needed
                pval_columns = [col for col in table.columns if col.endswith('_pval')]
                table.drop(columns=pval_columns, inplace=True)  # Use inplace=True

                # Update the original table in tables_and_effects
                if i == 0:
                    tab1 = table.copy()
                else:
                    tab2 = table.copy()





        if not conf_ints:
            tab1 = tab1[tab1['type'] != 'conf_int']
            tab2 = tab2[tab2['type'] != 'conf_int']

        tab2 = tab2.rename(columns={'var': 'x2'})
        values_cols = [col for col in tab2.columns if col not in ['x2', 'type']]


        tab22 = tab2.pivot(index='type', columns='x2', values=values_cols)
        tab22.columns = [f'{col[0]}_{col[1]}' for col in tab22.columns]
        tab22 = tab22.reset_index()


        table = pd.merge(tab1, tab22, on='type', how='inner')


        if percents:
            # Get full effect values for 'val' type rows
            val_rows = table[table['type'] == 'val']

            # Convert string values to numeric, removing asterisks
            numeric_cols = [col for col in table.columns if col not in ['type', 'var']]
            for col in numeric_cols:
                val_rows[col] = pd.to_numeric(val_rows[col].str.replace('*', ''), errors='coerce')

            # Create new rows with percentages
            percent_rows = val_rows.copy()
            percent_rows['type'] = 'percent'
            percent_rows['var'] = val_rows['var']

            # Calculate percentages
            for col in numeric_cols:
                percent_rows[col] = val_rows[col].div(val_rows['direct_effect']) * 100
                percent_rows[col] = percent_rows[col].apply(lambda x: f'({x:.0f}%)' if pd.notnull(x) else '')

            # Concatenate original table with percentage rows
            table = pd.concat([table, percent_rows], ignore_index=True)

        return table


    decomp_results = {}

    for dc in decomp_coefs:
        current_x1_vars = decomp_coefs + other_x1_vars

        if not groups:
            # Run the decomposition
            decomp_result = model.decompose_x1(
                x1_vars=current_x1_vars,
                decomp_var=dc,
                digits=decimals
            )


            full_tab = clean_tab(decomp_result)

        if groups:

            # We are taking the information from groups and parsing it
            combine_covariates = {}

            for group_name, variables in groups.items():
                # Handle case where variables is a single string
                if isinstance(variables, str):
                    variables = [variables]

                # Process each variable
                clean_vars = []
                for var in variables:
                    if var.startswith('C('):
                        # Extract the variable name from inside C()
                        base_var = var.split('(')[1].split(')')[0]
                        # Add pattern to match the transformed categorical variable format
                        clean_vars.append(f"C\\({base_var}\\)\\[T\\..*?\\]")
                    else:
                        # For regular variables, just escape any special characters
                        clean_vars.append(re.escape(var))

                # Create regex pattern by joining variables with '|'
                pattern = r".*(" + "|".join(clean_vars) + ").*"
                combine_covariates[group_name] = re.compile(pattern)

            # Run the decomposition
            decomp_result = model.decompose_x1(
                x1_vars=current_x1_vars,
                decomp_var=dc,
                digits=decimals,
                combine_covariates = combine_covariates
            )

            full_tab = clean_tab(decomp_result)
            # Store the result with the decomp_var as the key
        decomp_results[dc] = full_tab


    # Concatenate all tables in decomp_results
    final_table = pd.concat(decomp_results.values(), keys=decomp_results.keys(), ignore_index=False)

    other_cols = [col for col in final_table.columns if col not in ['var', 'type', 'direct_effect', 'full_effect', 'explained_effect']]
    # Reorder columns
    final_table = final_table[['var', 'type', 'direct_effect', 'full_effect', 'explained_effect'] + other_cols]

    # Only process where type is 'val'
    mask = final_table['type'] == 'val'
    # For val rows, extract the level
    final_table.loc[mask, 'var'] = (final_table.loc[mask, 'var']
                                   .str.extract(r'T\.([^]]+)')[0]
                                   .ffill()
                                   .combine_first(final_table.loc[mask, 'var']))  # Keep original if pattern not found
    if decomp_cat_mapping:
        # Create reverse mapping from table_var_name to display_name
        reverse_mapping = {v: k for k, v in decomp_cat_mapping.items()}
        final_table.loc[mask, 'var'] = final_table.loc[mask, 'var'].map(reverse_mapping).fillna(final_table.loc[mask, 'var'])

    # For non-val rows, set to empty string
    final_table.loc[~mask, 'var'] = ''

    final_table = final_table.reset_index(drop=True)

    type_mapping = {
        'val': 'Difference Estimate',
        'conf_int': 'Confidence Interval',
        'p_val': 'P Value',
        'percent': 'Percent of Total'
    }

    final_table['type'] = final_table['type'].replace(type_mapping)

    # Create the complete column mapping
    column_mapping = {
        'var': decomp_var_nicename,
        'type': '',
        'direct_effect': 'Total Difference',
        'full_effect': 'Unexplained Difference',
        'explained_effect': 'Explained Difference'
    }

    if not groups:
        # Add the explained difference columns using the x2_dict
        for nice_name, var_name in x2_dict.items():
            # Check if the variable is specified with C()
            if var_name.startswith('C('):
                # Extract the variable name from inside C()
                base_var = var_name.split('(')[1].split(')')[0]
                # Create pattern to match columns
                pattern = f'explained_effect_C\\({base_var}\\)\\[T\\..*?\\]'
                # Find matching columns
                matching_cols = [col for col in final_table.columns if re.match(pattern, col)]
                for col in matching_cols:
                    # Extract the level number
                    level = re.search(r'T\.([0-9]+)', col).group(1)
                    column_mapping[col] = f'Explained by {nice_name} = {level}'
            else:
                # Handle regular variables as before
                column_mapping[f'explained_effect_{var_name}'] = f'Explained by {nice_name}'

        # Apply all the renamings at once
        final_table = final_table.rename(columns=column_mapping)

    if groups:
        # Add any column that starts with 'explained_effect_'
        for col in final_table.columns:
            if col.startswith('explained_effect_'):
                # Get whatever comes after 'explained_effect_'
                remainder = col.replace('explained_effect_', '')
                column_mapping[col] = f'Explained by {remainder}'

        # Apply all the renamings at once
        final_table = final_table.rename(columns=column_mapping)

    #final_table = final_table.drop('type', axis=1)


    return final_table
