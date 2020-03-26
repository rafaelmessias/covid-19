from os.path import dirname, join
import math
from datetime import date, timedelta
from bokeh.models import *
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.plotting import figure
from bokeh.palettes import Category10_10 as colors
from bokeh.palettes import Blues9, Greys256
from bokeh.transform import dodge
import pandas as pd
import numpy as np
from scipy.spatial.distance import euclidean
from scipy.interpolate import interp1d

##### DATA

base_path = "COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_%s_global.csv"
data = {}
for name in ["confirmed", "deaths"]:
    data[name] = pd.read_csv(base_path % name)

# beds = pd.read_csv("beds.csv", sep=";", header=None)

#data["Confirmed"].head()
#beds.head()

##### GLOBALS




all_countries = sorted(data["confirmed"]["Country/Region"].unique())
# display_countries must be a list because the order is important
display_countries = [ "Italy", "Spain", "Sweden", "Denmark", "US" ]
# display_countries = ["Italy", "Spain", "US", "United Kingdom", "Germany" ]
country_sel_value = all_countries[0]
df_by_country = {}
# should_extrapolate = False
# extra_values = {}
# extrapolate_until_date = None
is_log_scale = True
align_by_values = [100, 10]
align_by = 0

##### CALLBACKS

def get_df_by_country():
    global df_by_country
    def get_df(country):
        def get_data(label):
            df = data[label]
            # Filter by country
            df = df[df["Country/Region"] == country]
            # Keep only numerical data
            df = df.iloc[:,4:]
            # Aggregate if there is more than one row
            df = df.sum(axis=0).to_frame()
            return df
        data_labels = ["confirmed", "deaths"]
        df = pd.concat([get_data(x) for x in data_labels], axis="columns")
        df.columns = data_labels
        return df

    df_by_country = {c:get_df(c) for c in display_countries}

    #df_by_country[countries[0]]

    # beds_by_country = {}
    # for c in countries:
    #     b = beds[beds[0] == c]
    #     beds_by_country[c] = int(b[1].iloc[0]) if len(b) > 0 else -1

    #beds_by_country

def align(df):
    column, value = ["confirmed", "deaths"][align_by], align_by_values[align_by]
    aligned_df = df[df[column] >= value]
    aligned_df.index = range(aligned_df.shape[0])
    return aligned_df


def make_figure():
    TOOLTIPS = [
        ("# of days", "@index"),
        ("Deaths", "@deaths"),
        ("Cases", "@confirmed")
    ]

    y_axis_type = "log" if is_log_scale else "linear"
    case_or_death = ["case", "death"][align_by]
    title = f"Aligned by the {align_by_values[align_by]}th {case_or_death} (dashed = deaths)"
    p = figure(title=title, plot_width=800, plot_height=400, tools="hover", tooltips=TOOLTIPS, 
            y_axis_type=y_axis_type, y_range = [10 ** 0, 10 ** 5])
    
    bar_width = 0.8 / len(df_by_country)

    for i, (country, df) in enumerate(df_by_country.items()):
        over100 = align(df)
        p.line("index", "confirmed", source=over100, line_width=2, legend_label=country, line_color=colors[i])
        p.line("index", "deaths", source=over100, line_width=2, legend_label=country, line_color=colors[i], line_dash='dashed')
        # p.vbar(x=dodge("index", -0.4+(i+1/2)*bar_width, range=p.x_range), top="Deaths", bottom=1, width=bar_width, source=over100, color=colors[i])
        p.circle("index", "confirmed", source=over100, line_width=2, color=colors[i], legend_label=country)
        # if should_extrapolate:
        #     print(">>> would extrapolate here:", country)
        #     extrap_df = extrapolate_until(country)
        #     p.line("index", "confirmed", source=extrap_df, line_width=2, line_color=colors[i], line_dash="dotted")

    p.legend.location = "top_left"
    p.xaxis.axis_label = "# of days after reaching 100 cases"
    p.yaxis.axis_label = "# of confirmed cases"

    return p


def make_heatmap():
    dist_matrix = np.zeros(shape=(len(display_countries), len(display_countries)))
    for i, c1 in enumerate(display_countries):
        df1_al = align(df_by_country[c1])["confirmed"]
        for j, c2 in enumerate(display_countries):
            df2_al = align(df_by_country[c2])["confirmed"]
            n = min(len(df1_al), len(df2_al))
            dist_matrix[i, j] = euclidean(df1_al[:n], df2_al[:n], [1/n]*n)
            # print(c1, c2, dist_matrix[i, j])

    factors = display_countries
    x, y = [x.ravel() for x in np.meshgrid(factors, factors, indexing='ij')]
    # cmap = np.array(viridis(256))
    cmap = np.array(Greys256)
    f = interp1d(np.linspace(dist_matrix.min(), dist_matrix.max(), len(cmap)), range(len(cmap)), kind='nearest')
    colors = cmap[f(dist_matrix.ravel()).astype(int)]

    hm = figure(title="Similarities (darker means more similar)", toolbar_location=None, plot_height=400, plot_width=400,
                x_range=factors, y_range=factors)

    hm.rect(x, y, color=colors, width=1, height=1)

    # return row(hm, dot, sizing_mode="scale_width")
    return hm


def make_figure_growth_rate():
    last_days = 10
    p = figure(title="Daily growth rate per country (day / day_before)", plot_width=800, plot_height=400)

    for i, (country, df) in enumerate(df_by_country.items()):
        df_latest = df.iloc[-last_days-1:,:]
        y = [df_latest["confirmed"].iloc[i] / df_latest["confirmed"].iloc[i-1] for i in range(1, len(df_latest))]
        p.line(x=range(len(y)), y=y, line_width=2, line_color=colors[i])
        p.circle(x=range(len(y)), y=y, color=colors[i])


    # p.legend.location = "top_left"
    p.xaxis.axis_label = f"last {last_days} days"
    p.yaxis.axis_label = "daily growth rate"

    return p


# def make_italy_comparison():
#     italy_al = align(df_by_country["Italy"])["Confirmed"]
#     dist_matrix = np.zeros(shape=(len(df_by_country),))
#     for i, (c1, df1) in enumerate(df_by_country.items()):
#         df1_al = align(df1)["Confirmed"]
#         n = min(len(df1_al), len(italy_al))
#         dist_matrix[i] = euclidean(df1_al[:n], italy_al[:n])
    
#     p = figure(title="Italy Comparison", plot_width=350, plot_height=350)
#     p.vbar(x=display_countries, top=dist_matrix, width=0.9)

#     return p


def country_sel_change(attrname, old, new):
    global country_sel_value
    country_sel_value = new

def country_add_click():
    # global display_countries
    if country_sel_value not in display_countries:
        display_countries.append(country_sel_value)
    update()

def country_rem_change(event):
    display_countries.remove(event.item)
    update()

# def extrapolate_until(c):
#     df = df_by_country[c]
#     aligned_conf = align(df)["confirmed"]        
#     # get average growth rate
#     rate = 0
#     for i in range(1, aligned_conf.shape[0]):
#         rate += aligned_conf.iloc[i] / aligned_conf.iloc[i-1]
#     rate = rate / (aligned_conf.shape[0] - 1)
#     # how many days to extrapolate for this country
#     extra = date.fromisoformat(extrapolate_until_date) - date.today()
#     new_values = [0] * extra.days
#     new_values[0] = aligned_conf.iloc[-1]
#     for i in range(1, len(new_values)):
#         new_values[i] = int(new_values[i-1] * rate)
#     new_index = range(len(aligned_conf) - 1, len(aligned_conf) + len(new_values) - 1)
#     return pd.DataFrame(new_values, index=new_index, columns=["confirmed"])


# def dt_pckr_change(attrname, old, new):
#     global should_extrapolate, extrapolate_until_date
#     should_extrapolate = True
#     extrapolate_until_date = new
#     update()


def checkbox_log_scale_click(event):
    global is_log_scale
    # print(event)
    is_log_scale = not is_log_scale
    update()


def align_by_radio_click(event):
    global align_by
    align_by = event
    update()

def align_by_text_change(attr, old, new):
    global align_by_value
    try:
        align_by_values[align_by] = int(new)
    except:
        pass
    update()
        
def make_controls():
    controls = []

    country_sel = Select(title='Countries:', value=country_sel_value, options=all_countries)
    country_sel.on_change('value', country_sel_change)
    controls.append(country_sel)

    country_add_btn = Button(label="Add", button_type="success")
    country_add_btn.on_click(country_add_click)
    controls.append(country_add_btn)

    menu = [(x, x) for x in display_countries]
    country_rem_ddo = Dropdown(label="Remove", button_type="warning", menu=menu)
    country_rem_ddo.on_click(country_rem_change)
    controls.append(country_rem_ddo)

    # dt_pckr_value = extrapolate_until_date if extrapolate_until_date else date.today()
    # dt_pckr_strt = DatePicker(title='Extrapolate until:', value=dt_pckr_value, min_date=date.today(), max_date=date.today() + timedelta(20))
    # dt_pckr_strt.on_change('value', dt_pckr_change)
    # controls.append(dt_pckr_strt)

    active = [0] if is_log_scale else []
    checkbox_log_scale = CheckboxGroup(labels=["Log scale"], active=active)
    checkbox_log_scale.on_click(checkbox_log_scale_click)
    controls.append(checkbox_log_scale)

    align_by_title = Div(text="Align by:")
    align_by_radio = RadioButtonGroup(
        labels=["Confirmed", "Deaths"], active=align_by)
    align_by_radio.on_click(align_by_radio_click)
    align_by_text = TextInput(value=str(align_by_values[align_by]))
    align_by_text.on_change("value", align_by_text_change)
    controls.append(align_by_title)
    controls.append(align_by_text)    
    controls.append(align_by_radio)

    return column(*controls, width=200)

def example():
    import numpy as np

    from bokeh.models import ColorBar, LogColorMapper, LogTicker
    from bokeh.plotting import figure, output_file, show
    

    def normal2d(X, Y, sigx=1.0, sigy=1.0, mux=0.0, muy=0.0):
        z = (X-mux)**2 / sigx**2 + (Y-muy)**2 / sigy**2
        return np.exp(-z/2) / (2 * np.pi * sigx * sigy)

    X, Y = np.mgrid[-3:3:100j, -2:2:100j]
    Z = normal2d(X, Y, 0.1, 0.2, 1.0, 1.0) + 0.1*normal2d(X, Y, 1.0, 1.0)
    image = Z * 1e6

    color_mapper = LogColorMapper(palette="Viridis256", low=1, high=1e7)

    plot = figure(x_range=(0,1), y_range=(0,1), toolbar_location=None)
    plot.image(image=[image], color_mapper=color_mapper,
            dh=[1.0], dw=[1.0], x=[0], y=[0])

    color_bar = ColorBar(color_mapper=color_mapper, ticker=LogTicker(),
                        label_standoff=12, border_line_color=None, location=(0,0))

    plot.add_layout(color_bar, 'right')

    return plot


def update():
    get_df_by_country()
    layout.children[0] = make_controls()
    layout.children[1] = column(
        row(make_figure(), make_heatmap()),
        make_figure_growth_rate()
    )

##### LAYOUT

get_df_by_country()
layout = row(
    make_controls(), 
    column(
        row(make_figure(), make_heatmap()),
        make_figure_growth_rate())
)

curdoc().add_root(layout)
curdoc().title = "COVID-19"
