import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve

from dash import Dash, html, dcc, callback, Output, Input
import plotly.express as px

def load_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.normpath(
        os.path.join(base_dir, "..", "..", "data", "Speed Dating Data.csv")
    )
    speed = pd.read_csv(data_path, encoding="latin1")

    features = ["age", "age_o", "samerace", "int_corr", "attr", "intel", "fun", "amb"]
    target = "match"
    df_model = speed[features + [target]].dropna()

    # Feature Engineering
    df_model["age_diff"] = abs(df_model["age"] - df_model["age_o"])
    df_model = df_model.drop(columns=["age", "age_o"])

    X = df_model.drop(columns=[target])
    y = df_model[target]
    return X, y

X, y = load_data()

app = Dash(__name__)

app.layout = [
    html.H1(children="Predictive Model Dashboard", style={"textAlign": "center"}),
    html.P("Please input numbers only for each item:"),
    html.P(
        "The values, from top to bottom, are age difference, number of shared interests, how important a potential partner's attractiveness is, how important a potential partner's intelligence is, how important a potential partner's fun-loving level is, how important a potential partner's ambition is, and if you and your partner are the same race."
    ),
    html.P(
        "Age difference accepts values from 0 to 60. Shared interests goes from 0 to 20. Attractiveness, intelligence, fun, and ambition accepts values from 0 to 10 (0 is least important, 10 is most important). Same race accepts 0 (not the same) or 1 (same)."
    ),
    dcc.Input(value=5.0, type="number", id="age_diff", min=0.0, max=60.0, step=1.0),
    dcc.Input(value=5.0, type="number", id="int_corr", min=0.0, max=20.0, step=1.0),
    dcc.Input(value=5.0, type="number", id="attr", min=0.0, max=10.0, step=1.0),
    dcc.Input(value=5.0, type="number", id="intel", min=0.0, max=10.0, step=1.0),
    dcc.Input(value=5.0, type="number", id="fun", min=0.0, max=10.0, step=1.0),
    dcc.Input(value=5.0, type="number", id="amb", min=0.0, max=10.0, step=1.0),
    dcc.Input(value=0.0, type="number", id="samerace", min=0.0, max=1.0, step=1.0),
    html.P("Select Model:"),
    dcc.Dropdown(
        id="dropdown",
        options=["Logistic Regression", "Random Forest"],
        value="Logistic Regression",
        clearable=False,
    ),
    html.H4(children="Predicted Group from Text Input:"),
    html.Div(id="result"),
    html.H4(children="ROC Curve"),
    dcc.Graph(id="graph"),
]

@callback(
    Output("graph", "figure"),
    Output("result", "children"),
    Input("dropdown", "value"),
    Input("age_diff", "value"),
    Input("int_corr", "value"),
    Input("attr", "value"),
    Input("intel", "value"),
    Input("fun", "value"),
    Input("amb", "value"),
    Input("samerace", "value"),
)
def train_and_display(model_name, age_diff, int_corr, attr, intel, fun, amb, samerace):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    if model_name == "Random Forest":
        model = RandomForestClassifier(
            max_depth=5, min_samples_leaf=1, min_samples_split=2, n_estimators=200
        )
    if model_name == "Logistic Regression":
        model = LogisticRegression(max_iter=1000)

    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)

    fig = px.area(
        x=fpr,
        y=tpr,
        title=f"ROC Curve (AUC={auc:.4f})",
        labels=dict(x="False Positive Rate", y="True Positive Rate"),
    )
    fig.add_shape(type="line", line=dict(dash="dash"), x0=0, x1=1, y0=0, y1=1)

    input_dict = {
        "samerace": [samerace],
        "int_corr": [int_corr],
        "attr": [attr],
        "intel": [intel],
        "fun": [fun],
        "amb": [amb],
        "age_diff": [age_diff],
    }
    input_df = pd.DataFrame(input_dict)
    output_pred = model.predict(input_df)

    return fig, output_pred

def run_dashboard():
    app.run(debug=True)

if __name__ == "__main__":
    run_dashboard()
