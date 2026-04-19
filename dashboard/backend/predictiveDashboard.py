import os
import joblib
import pandas as pd

from dash import Dash, html, dcc, callback, Output, Input
import plotly.graph_objects as go

base_dir = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.normpath(
    os.path.join(base_dir, "..", "..", "Model", "match_model.joblib")
)

model = joblib.load(model_path) #Had assistance from AI (ChatGPT) for loading model using joblib

FEATURE_ORDER = ["samerace", "int_corr", "attr", "intel", "fun", "amb", "age_diff"]

app = Dash(__name__)

app.layout = html.Div(
    [
        html.H1("Predictive Model Dashboard", style={"textAlign": "center"}),
    
        html.P(
            "This is a predictive model that takes input representing a relationship between two users, for example same race is if they are the same race, age difference is calculated from each individual's age, shared interest correlation is calculated from each individual's interests etc, the rest is one person's perception of the other, and predicts the probability that they would match on a dating application.",
            style={"textAlign": "center"}
        ),

        html.P(
            "Enter values for the features below to estimate match compatibility.",
            style={"textAlign": "center"}
        ),

        html.P(
            "Age difference is the absolute age gap between two people. "
            "Shared interest correlation ranges from -1 to 1. "
            "Attractiveness, intelligence, fun, and ambition ratings range from 1 to 10. "
            "Same race accepts 0 (not same) or 1 (same)."
        ),

        html.Br(),

        html.Label("Age Difference"),
        dcc.Input(
            id="age_diff",
            value=5.0,
            type="number",
            min=0.0,
            max=60.0,
            step=1.0,
            style={"width": "100%"}
        ),

        html.Br(), html.Br(),

        html.Label("Shared Interest Correlation"),
        dcc.Input(
            id="int_corr",
            value=0.0,
            type="number",
            min=-1.0,
            max=1.0,
            step=0.05,
            style={"width": "100%"}
        ),

        html.Br(), html.Br(),

        html.Label("Attractiveness Rating"),
        dcc.Input(
            id="attr",
            value=5.0,
            type="number",
            min=1.0,
            max=10.0,
            step=0.5,
            style={"width": "100%"}
        ),

        html.Br(), html.Br(),

        html.Label("Intelligence Rating"),
        dcc.Input(
            id="intel",
            value=5.0,
            type="number",
            min=1.0,
            max=10.0,
            step=0.5,
            style={"width": "100%"}
        ),

        html.Br(), html.Br(),

        html.Label("Fun Rating"),
        dcc.Input(
            id="fun",
            value=5.0,
            type="number",
            min=1.0,
            max=10.0,
            step=0.5,
            style={"width": "100%"}
        ),

        html.Br(), html.Br(),

        html.Label("Ambition Rating"),
        dcc.Input(
            id="amb",
            value=5.0,
            type="number",
            min=1.0,
            max=10.0,
            step=0.5,
            style={"width": "100%"}
        ),

        html.Br(), html.Br(),

        html.Label("Same Race (0 = No, 1 = Yes)"),
        dcc.Input(
            id="samerace",
            value=0,
            type="number",
            min=0,
            max=1,
            step=1,
            style={"width": "100%"}
        ),

        html.Br(), html.Br(),

        html.H4("Prediction"),
        html.Div(id="result", style={"fontSize": "28px", "fontWeight": "bold"}),

        html.Br(),

        html.H4("Predicted Match Probability"),
        html.Div(id="probability", style={"fontSize": "24px"}),

        html.Br(),

        html.H4("Interpretation"),
        html.Div(id="interpretation", style={"fontSize": "18px"}),

        html.Br(),

        html.H4("Feature Inputs Used"),
        html.Div(id="feature_table", style={"fontSize": "18px"}),

        html.Br(),

        html.H4("Model Note"),
        html.Div(
            id="disclaimer",
            style={"fontSize": "16px", "fontStyle": "italic", "color": "#555"}
        ),

        html.Br(),

        dcc.Graph(id="placeholder_graph"),
    ],
    style={"maxWidth": "1100px", "margin": "0 auto", "padding": "30px"},
)

@callback(
    Output("result", "children"),
    Output("probability", "children"),
    Output("interpretation", "children"),
    Output("feature_table", "children"),
    Output("disclaimer", "children"),
    Output("placeholder_graph", "figure"),
    Input("age_diff", "value"),
    Input("int_corr", "value"),
    Input("attr", "value"),
    Input("intel", "value"),
    Input("fun", "value"),
    Input("amb", "value"),
    Input("samerace", "value"),
)
def predict_match(age_diff, int_corr, attr, intel, fun, amb, samerace):
    input_df = pd.DataFrame([{
        "samerace": samerace,
        "int_corr": int_corr,
        "attr": attr,
        "intel": intel,
        "fun": fun,
        "amb": amb,
        "age_diff": age_diff,
    }])[FEATURE_ORDER]

    pred_num = int(model.predict(input_df)[0])
    pred_prob = float(model.predict_proba(input_df)[0, 1])

    if pred_prob < 0.30:
        result_text = html.Span("Low likelihood of match", style={"color": "#b02a37"})
    elif pred_prob < 0.70:
        result_text = html.Span("Moderate compatibility", style={"color": "#997404"})
    else:
        result_text = html.Span("High compatibility", style={"color": "#146c43"})

    prob_text = f"{pred_prob:.1%}"

    interpretation_text = (
        "This estimate is primarily influenced by fun and attractiveness, "
        "which were the most important features in our trained models. "
        "Shared interests and intelligence also contribute, while homophily-related "
        "features such as same race and age difference had smaller effects."
    )

    feature_rows = html.Ul([
        html.Li(f"{col}: {input_df.iloc[0][col]}") for col in FEATURE_ORDER
    ])

    disclaimer_text = (
        "This prediction is based on patterns learned from historical speed dating data "
        "and should be interpreted as a probabilistic estimate rather than a deterministic outcome."
    )

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=pred_prob * 100,
            title={"text": "Match Probability (%)"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"thickness": 0.3},
                "steps": [
                    {"range": [0, 30], "color": "#f8d7da"},
                    {"range": [30, 70], "color": "#fff3cd"},
                    {"range": [70, 100], "color": "#d1e7dd"},
                ],
            },
        )
    )
    fig.update_layout(height=400)

    return (
        result_text,
        prob_text,
        interpretation_text,
        feature_rows,
        disclaimer_text,
        fig,
    )

def run_dashboard():
    app.run(debug=True)

if __name__ == "__main__":
    run_dashboard()