import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve

from dash import Dash, html, dcc, callback, Output, Input
import plotly.express as px

speed = pd.read_csv("Capstone_Team15/data/Speed Dating Data.csv", encoding="latin1")

features = ["age", "age_o", "samerace", "int_corr", "attr", "intel", "fun", "amb"]

target = "match"
df_model = speed[features + [target]].dropna()

# Feature Engineering
df_model["age_diff"] = abs(df_model["age"] - df_model["age_o"])
df_model = df_model.drop(columns=["age", "age_o"])

X = df_model.drop(columns=[target])
y = df_model[target]

app = Dash()

MODELS = {
    "Logistic Regression": LogisticRegression,
    "Random Forest": RandomForestClassifier,
}

# Requires Dash 2.17.0 or later
app.layout = [
    html.H1(children="Predictive Model Dashboard", style={"textAlign": "center"}),
    dcc.Input(value=5.0, type="number", id="age_diff", min=0.0, max=60.0, step=1.0),
    dcc.Input(value=5.0, type="number", id="int_corr", min=0.0, max=10.0, step=1.0),
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

    # y_pred = model.predict(X_test)
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


if __name__ == "__main__":
    app.run(debug=True)
