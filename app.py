from flask import Flask,render_template,request
import pandas as pd

from sklearn.model_selection import train_test_split


app=Flask(__name__)
@app.route("/",methods=["GET","POST"])
def index():
    return render_template("index.html")

@app.route("/data",methods=["GET","POST"])
def data():
    if request.method=="POST":
        f = request.files["csvfile"]
        data = pd.read_csv(f)
        x = data.iloc[:, :-1]
        y = data.iloc[:, -1]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=0)
        algoritm = request.form["select an algoritm"]
        if algoritm == "linear regression":

            from sklearn.linear_model import LinearRegression
            lr = LinearRegression()
            lr.fit(x_train, y_train)
            y_pred = lr.predict(x_test)
            from sklearn.metrics import r2_score
            c = r2_score(y_test, y_pred)
            return render_template("final.html", data=c)

        if algoritm == "decision tree regressor":


            from sklearn.tree import DecisionTreeRegressor
            regressor = DecisionTreeRegressor( random_state=0)

            regressor.fit(x_train, y_train)
            y_pred = regressor.predict(x_test)
            from sklearn.metrics import r2_score
            c = r2_score(y_test, y_pred)
            return render_template("final.html", data=c)
        if algoritm == "random forest regressor":
            from sklearn.ensemble import RandomForestRegressor
            regressor = RandomForestRegressor(n_estimators=100, random_state=0)
            regressor.fit(x_train, y_train)
            y_pred = regressor.predict(x_test)
            from sklearn.metrics import r2_score
            c = r2_score(y_test, y_pred)
            return render_template("final.html", data=c)

if __name__=="__main__":
    app.run(debug=True)
