# Imports from 3rd party libraries
import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import pandas as pd
from joblib import load
import shap
import matplotlib.pyplot as plt
import io
import base64

# Imports from this application
from app import app

pipeline = load('assets/pipeline.joblib')

# 2 column layout. 1st column width = 4/12
# https://dash-bootstrap-components.opensource.faculty.ai/l/components/layout
column1 = dbc.Col(
    [
        dcc.Markdown('#### Requested Loan Amount'), 
        dcc.Slider(
            id='loan_amnt', 
            min=1000, 
            max=40000, 
            step=500, 
            value=20000, 
            marks={n: str(n) for n in range(1000,40000,5000)}, 
            className='mb-5', 
        ), 
        dcc.Markdown('#### Interest Rate'), 
        dcc.Slider(
            id='int_rate', 
            min=7, 
            max=36, 
            step=0.25, 
            value=15, 
            marks={n: str(n) for n in range(5,40,5)}, 
            className='mb-5', 
        ), 
        dcc.Markdown('#### Term'), 
        dcc.Dropdown(
            id='term', 
            options = [
                {'label': '36 months', 'value': '36 months'}, 
                {'label': '60 months', 'value': '60 months'}, 
            ], 
            value = '36 months', 
            className='mb-5', 
        ), 
    ],
    md=6,
)

column2 = dbc.Col(
    [
        dcc.Markdown('#### Fico Credit Score'), 
        dcc.Slider(
            id='fico_range_high', 
            min=300, 
            max=850, 
            step=10, 
            value=700, 
            marks={n: str(n) for n in range(300,850,50)}, 
            className='mb-5', 
        ), 
        dcc.Markdown('#### Annual Income'), 
        dcc.Slider(
            id='annual_inc', 
            min=10000, 
            max=200000, 
            step=1000, 
            value=50000, 
            marks={n: str(n) for n in range(10000,200000,20000)}, 
            className='mb-5', 
        ), 
        dcc.Markdown('#### Home Ownership'), 
        dcc.Dropdown(
            id='home_ownership', 
            options = [
                {'label': 'Mortgage', 'value': 'MORTGAGE'}, 
                {'label': 'Rent', 'value': 'RENT'}, 
                {'label': 'Own (No mortgage)', 'value': 'OWN'}, 
            ], 
            value = 'RENT', 
            className='mb-5', 
        ), 
    ],
    md=6,
)

column3 = dbc.Col(
    [
        html.H2('Predicted Loan Risk', className='mb-5'), 
        html.Div(id='prediction-content', className='lead'),
        html.Button('Explain Prediction', id='explain-btn'),
        html.Div([html.Img(id='shap-img', height=200, width=1000)])
    ]
)

layout = html.Div(
    [
        dbc.Row([column1, column2]),
        dbc.Row(column3)
    ]
)

@app.callback(
    Output('prediction-content', 'children'),
    [Input('loan_amnt', 'value'), 
     Input('int_rate', 'value'),
     Input('term', 'value'),
     Input('fico_range_high', 'value'),
     Input('annual_inc', 'value'),
     Input('home_ownership', 'value')],
)
def predict(loan_amnt, int_rate, term, fico_range_high, annual_inc, home_ownership):   
    # Convert input to dataframe
    df = pd.DataFrame(
        data=[[loan_amnt, int_rate, term, fico_range_high, annual_inc, home_ownership]],
        columns=['loan_amnt', 'int_rate', 'term', 'fico_range_high', 'annual_inc', 'home_ownership']
    )

    # Make predictions (includes predicted probability)
    pred_proba = pipeline.predict_proba(df)[0, 1]
    pred_proba *= 100

    # Show predictiion & probability
    return (f'The model predicts this loan has a {pred_proba:.0f}% probability of being Fully Paid.')

@app.callback(
    Output('shap-img', 'src_image'),
    [Input('explain-btn','n_clicks')],
    [State('loan_amnt', 'value'), 
     State('int_rate', 'value'),
     State('term', 'value'),
     State('fico_range_high', 'value'),
     State('annual_inc', 'value'),
     State('home_ownership', 'value')],
)
def explain_png(n_clicks, loan_amnt, int_rate, term, fico_range_high, annual_inc, home_ownership):
    
    # Convert input to dataframe
    df = pd.DataFrame(
        data=[[loan_amnt, int_rate, term, fico_range_high, annual_inc, home_ownership]],
        columns=['loan_amnt', 'int_rate', 'term', 'fico_range_high', 'annual_inc', 'home_ownership']
    )

    # Get steps from pipeline and transform
    model = pipeline.named_steps['xgbclassifier']
    encoder = pipeline.named_steps['ordinalencoder']
    df_processed = encoder.transform(df)
    
    # Get shapley additive explanations
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(df_processed)

    # Plot shapley and save matplotlib plot to base64 encoded buffer for rendering
    fig = shap.force_plot(
        base_value=explainer.expected_value, 
        shap_values=shap_values, 
        features=df_processed, 
        link='logit',
        show=True,
        matplotlib=True)
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("ascii").replace("\n", "")
    src_image = "data:image/png;base64,{}".format(encoded)

    # Return image
    return src_image