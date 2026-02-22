## Causal Inference for the Brave and True


```python
import pandas as pd

import pyfixest as pf
```

### Chapter 14: Panel Data and Fixed Effects

In this example we replicate the results of the great (freely available reference!) Causal Inference for the Brave and True - Chapter 14. Please refer to the original text for a detailed explanation of the data.


```python
data_path = "https://raw.githubusercontent.com/bashtage/linearmodels/main/linearmodels/datasets/wage_panel/wage_panel.csv.bz2"
data_df = pd.read_csv(data_path)

data_df.head()
```

```text
   nr  year  black  exper  hisp  hours  married  educ  union     lwage  \
0  13  1980      0      1     0   2672        0    14      0  1.197540   
1  13  1981      0      2     0   2320        0    14      1  1.853060   
2  13  1982      0      3     0   2940        0    14      0  1.344462   
3  13  1983      0      4     0   2960        0    14      0  1.433213   
4  13  1984      0      5     0   3071        0    14      0  1.568125   

   expersq  occupation  
0        1           9  
1        4           9  
2        9           9  
3       16           9  
4       25           5  
```

We have a classical panel data set with units (nr) and time (year).

We are interested in estimating the effect of marriage status on log wage, using a set of controls (union, hours) and individual (nr) and year fixed effects. 

```python
panel_fit = pf.feols(
    fml="lwage ~ married + expersq + union + hours | nr + year",
    data=data_df,
    vcov={"CRV1": "nr + year"},
    demeaner_backend="rust",
)
```

```python
pf.etable(panel_fit)
```

```text
GT(_tbl_data=  level_0               level_1                       0
0    coef               married     0.048* <br> (0.018)
1    coef               expersq  -0.006*** <br> (0.001)
2    coef                 union     0.073* <br> (0.023)
3    coef                 hours   -0.000** <br> (0.000)
4      fe                    nr                       x
5      fe                  year                       x
6   stats          Observations                    4360
7   stats             S.E. type             by: nr+year
8   stats         R<sup>2</sup>                   0.631
9   stats  R<sup>2</sup> Within                   0.047, _body=<great_tables._gt_data.Body object at 0x000001E546E22C30>, _boxhead=Boxhead([ColInfo(var='level_0', type=<ColInfoTypeEnum.row_group: 3>, column_label='level_0', column_align='center', column_width=None), ColInfo(var='level_1', type=<ColInfoTypeEnum.stub: 2>, column_label='level_1', column_align='center', column_width=None), ColInfo(var='0', type=<ColInfoTypeEnum.default: 1>, column_label='(1)', column_align='center', column_width=None)]), _stub=<great_tables._gt_data.Stub object at 0x000001E546115DF0>, _spanners=Spanners([SpannerInfo(spanner_id='lwage', spanner_level=1, spanner_label='lwage', spanner_units=None, spanner_pattern=None, vars=['0'], built=None)]), _heading=Heading(title=None, subtitle=None, preheader=None), _stubhead=None, _source_notes=['Significance levels: * p < 0.05, ** p < 0.01, *** p < 0.001. Format of coefficient cell:\nCoefficient \n (Std. Error)'], _footnotes=[], _styles=[], _locale=<great_tables._gt_data.Locale object at 0x000001E546EB2420>, _formats=[], _substitutions=[], _options=Options(table_id=OptionsInfo(scss=False, category='table', type='value', value=None), table_caption=OptionsInfo(scss=False, category='table', type='value', value=None), table_width=OptionsInfo(scss=True, category='table', type='px', value='auto'), table_layout=OptionsInfo(scss=True, category='table', type='value', value='fixed'), table_margin_left=OptionsInfo(scss=True, category='table', type='px', value='auto'), table_margin_right=OptionsInfo(scss=True, category='table', type='px', value='auto'), table_background_color=OptionsInfo(scss=True, category='table', type='value', value='#FFFFFF'), table_additional_css=OptionsInfo(scss=False, category='table', type='values', value=[]), table_font_names=OptionsInfo(scss=False, category='table', type='values', value=['-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'Roboto', 'Oxygen', 'Ubuntu', 'Cantarell', 'Helvetica Neue', 'Fira Sans', 'Droid Sans', 'Arial', 'sans-serif']), table_font_size=OptionsInfo(scss=True, category='table', type='px', value='16px'), table_font_weight=OptionsInfo(scss=True, category='table', type='value', value='normal'), table_font_style=OptionsInfo(scss=True, category='table', type='value', value='normal'), table_font_color=OptionsInfo(scss=True, category='table', type='value', value='#333333'), table_font_color_light=OptionsInfo(scss=True, category='table', type='value', value='#FFFFFF'), table_border_top_include=OptionsInfo(scss=False, category='table', type='boolean', value=True), table_border_top_style=OptionsInfo(scss=True, category='table', type='value', value='solid'), table_border_top_width=OptionsInfo(scss=True, category='table', type='px', value='2px'), table_border_top_color=OptionsInfo(scss=True, category='table', type='value', value='#A8A8A8'), table_border_right_style=OptionsInfo(scss=True, category='table', type='value', value='none'), table_border_right_width=OptionsInfo(scss=True, category='table', type='px', value='2px'), table_border_right_color=OptionsInfo(scss=True, category='table', type='value', value='#D3D3D3'), table_border_bottom_include=OptionsInfo(scss=False, category='table', type='boolean', value=True), table_border_bottom_style=OptionsInfo(scss=True, category='table', type='value', value='hidden'), table_border_bottom_width=OptionsInfo(scss=True, category='table', type='px', value='2px'), table_border_bo
...[truncated]...
```

We obtain the same results as in the book!